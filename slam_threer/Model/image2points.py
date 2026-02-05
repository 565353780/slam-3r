import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12

from slam_threer.Model.blocks import Mlp
from slam_threer.Model.heads.postprocess import reg_dense_conf
from slam_threer.Model.multiview_3d import Multiview3D
from slam_threer.Method.device import MyNvtxRange


class Image2PointsModel(Multiview3D):
    """Image2Point Model, with a retrieval module attached to it.
    Take multiple views as input, and recover 3D pointmaps directly. 
    All the pointmaps are in the coordinate system of a designated view.
    """
    def __init__(self, corr_depth=2, **args):
        super().__init__( **args)
        self.corr_score_depth = corr_depth
        self.corr_score_norm = nn.LayerNorm(self.dec_embed_dim)
        self.corr_score_proj = Mlp(in_features=self.dec_embed_dim, out_features=1)

    def get_corr_score(self, views, ref_id, depth=-1):
        """Get the correlation score between the reference view and each source view
        Use the first serveral decoder blocks, followed by a layernorm and a mlp. 
        Modified from _decode_multiview() function.
        
        Args:
            ref_id: index of the reference view
            depth: number of decoder blocks to use. If -1, use self.corr_score_depth
        
        Returns:
            patch_corr_scores: correlation scores between the reference view 
            and each source view tokens
        
        """
        if depth < 0:
            depth = self.corr_score_depth
        shapes, enc_feats, poses = self._encode_multiview(views)
        assert ref_id < len(views) and ref_id >= 0
        src_ids = [i for i in range(len(views)) if i != ref_id]
        ref_ids = [ref_id] 
        
        # select and stacck up ref and src elements. R=1
        ref_feats, src_feats = self.split_stack_ref_src(enc_feats, ref_ids, src_ids) # (R, B, S, D), (V-R, B, S, D)
        ref_poses, src_poses = self.split_stack_ref_src(poses, ref_ids, src_ids)  # (R, B, S, 2), (V-R, B, S, 2)
        
        num_ref = ref_feats.shape[0]
        num_src = src_feats.shape[0]
        num_views = num_ref + num_src
        
        final_refs = [ref_feats]  # before projection
        final_srcs = [src_feats]
        # project to decoder dim
        final_refs.append(self.decoder_embed(ref_feats))
        final_srcs.append(self.decoder_embed(src_feats))
        
        ref_rel_ids_d = torch.arange(num_views-1, device=final_refs[0].device, dtype=torch.long)
        src_rel_ids_d = torch.zeros(num_views-1, device=final_srcs[0].device, dtype=torch.long)  
        
        for i in range(depth):
            ref_input = final_refs[-1]  # (1, B, S, D)
            src_inputs = final_srcs[-1]  # (V-1, B, S, D)

            ref_blk = self.mv_dec_blocks1[i]
            src_blk = self.mv_dec_blocks2[i]
            # reference image side
            if i < depth-1:
                ref_outputs = ref_blk(ref_input, src_inputs, 
                                        ref_poses, src_poses, 
                                        ref_rel_ids_d, num_views-1)
                final_refs.append(ref_outputs)
            # source image side
            src_outputs = src_blk(src_inputs, ref_input, 
                                     src_poses, ref_poses, 
                                     src_rel_ids_d, 1)
            final_srcs.append(src_outputs)

        dec_feats_shallow = final_srcs[-1] #output of the depth_th block (src, B, S, D)
        dec_feats_shallow = self.corr_score_norm(dec_feats_shallow)
        patch_corr_scores = self.corr_score_proj(dec_feats_shallow)[..., 0]  # (src, B, S)
        patch_corr_scores = reg_dense_conf(patch_corr_scores, mode=self.conf_mode)  # (src, B, S)   

        return patch_corr_scores
    
    def forward(self, views:list, ref_id, return_corr_score=False):
        """ 
        naming convention:
            reference views: views that define the coordinate system.
            source views: views that need to be transformed to the coordinate system of the reference views.
        Args:
            views: list of dictionaries, each containing:
                - 'img': input image tensor (B, 3, H, W) or 'img_tokens': image tokens (B, S, D)
                - 'true_shape': true shape of the input image (B, 2)
            ref_id: index of the reference view in input view list
        """
        # decide which views are reference views and which are source views
        assert ref_id < len(views) and ref_id >= 0
        src_ids = [i for i in range(len(views)) if i != ref_id]
        ref_ids = [ref_id] 
            
        with MyNvtxRange('encode'):
            shapes, enc_feats, poses = self._encode_multiview(views)
        
        # select and stacck up ref and src elements. R=1 in the I2P model.
        ref_feats, src_feats = self.split_stack_ref_src(enc_feats, ref_ids, src_ids) # (R, B, S, D), (V-R, B, S, D)
        ref_poses, src_poses = self.split_stack_ref_src(poses, ref_ids, src_ids)  # (R, B, S, 2), (V-R, B, S, 2)
        ref_shapes, src_shapes = self.split_stack_ref_src(shapes, ref_ids, src_ids) # (R, B, 2), (V-R, B, 2)
        
        # let all reference view and source view tokens interact with each other
        with MyNvtxRange('decode'):
            dec_feats_ref, dec_feats_src = self._decode_multiview(ref_feats, src_feats, 
                                                                  ref_poses, src_poses,
                                                                  None,None) 
        # print(len(dec_feats_ref), len(dec_feats_src)) #list: [depth*(R*B, S, D/D')], [depth*((V-R)*B, S, D/D')]
        
        with MyNvtxRange('head'):
            with torch.cuda.amp.autocast(enabled=False):
                # conf: ((V-R)*B, H, W)  pts3d: ((V-R)*B, H, W, 3)
                res_ref = self._downstream_head(1, [tok.float() for tok in dec_feats_ref], ref_shapes.reshape(-1,2))
                res_src = self._downstream_head(2, [tok.float() for tok in dec_feats_src], src_shapes.reshape(-1,2))
                # print(res_ref['pts3d'].shape, res_src['pts3d'].shape, res_ref['conf'].shape, res_src['conf'].shape)
        
        if return_corr_score:
            dec_feats_shallow = dec_feats_src[self.corr_score_depth] # (src*B, S, D)
            dec_feats_shallow = self.corr_score_norm(dec_feats_shallow)
            patch_corr_scores = self.corr_score_proj(dec_feats_shallow)[..., 0]  # (src*B, S)
            # patch_confs = reg_dense_conf(patch_confs, mode=self.conf_mode)  # (src*B, S)        
        
        # split the results back to each view
        results = [] 
        B = res_ref['pts3d'].shape[0]  #因为这里num_ref=1
        for id in range(len(views)):
            res = {}
            if id in ref_ids:
                rel_id = ref_ids.index(id)
                res['pts3d'] = res_ref['pts3d'][rel_id*B:(rel_id+1)*B]
                res['conf'] = res_ref['conf'][rel_id*B:(rel_id+1)*B]
            else:
                rel_id = src_ids.index(id)
                res['pts3d_in_other_view'] = res_src['pts3d'][rel_id*B:(rel_id+1)*B]
                res['conf'] = res_src['conf'][rel_id*B:(rel_id+1)*B] # (B, H, W)
                if return_corr_score:
                    res['pseudo_conf'] = patch_corr_scores[rel_id*B:(rel_id+1)*B] # (B, S)
            results.append(res)
        return results
 
