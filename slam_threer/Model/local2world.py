import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12

from slam_threer.Model.multiview_3d import Multiview3D
from slam_threer.Method.device import MyNvtxRange


class Local2WorldModel(Multiview3D):
    """Local2World Model
    Take arbitrary number of refernce views('scene frames' in paper) 
    and source views('keyframes' in paper) as input
    1. refine the input 3D pointmaps of the reference views
    2. transform the input 3D pointmaps of the source views to the coordinate system of the reference views
    """
    def __init__(self, **args):
        super().__init__(**args)
        self.dec_embed_dim = self.decoder_embed.out_features
        self.void_pe_token = nn.Parameter(torch.randn(1,1,self.dec_embed_dim), requires_grad=True)
        self.set_pointmap_embedder()

    def set_pointmap_embedder(self):
        self.ponit_embedder = nn.Conv2d(3, self.dec_embed_dim, 
                                        kernel_size=self.patch_size, stride=self.patch_size)
        
    def get_pe(self, views, ref_ids):
        """embed 3D points with a single conv layer
        landscape_only not tested yet"""
        pes = []
        for id, view in enumerate(views):
            if id in ref_ids:
                pos = view['pts3d_world']
            else:
                pos = view['pts3d_cam']
                
            if pos.shape[-1] == 3:
                pos = pos.permute(0,3, 1, 2)
                
            pts_embedding = self.ponit_embedder(pos).permute(0,2,3,1).reshape(pos.shape[0], -1, self.dec_embed_dim) # (B, S, D)
            if 'patch_mask' in view:
                patch_mask = view['patch_mask'].reshape(pos.shape[0], -1, 1) # (B, S, 1)
                pts_embedding = pts_embedding*(~patch_mask) + self.void_pe_token*patch_mask
                
            pes.append(pts_embedding)
        
        return pes
    
    def forward(self, views:list, ref_ids = 0):
        """ 
        naming convention:
            reference views: views that define the coordinate system.
            source views: views that need to be transformed to the coordinate system of the reference views.
        
        Args:
            views: list of dictionaries, each containing:
                    - 'img': input image tensor (B, 3, H, W) or 'img_tokens': image tokens (B, S, D)
                    - 'true_shape': true shape of the input image (B, 2)
                    - 'pts3d_world' (reference view only): 3D pointmaps in the world coordinate system (B, H, W, 3)
                    - 'pts3d_cam' (source view only): 3D pointmaps in the camera coordinate system (B, H, W, 3)
            ref_ids: indexes of the reference views in the input view list
        """
        # decide which views are reference views and which are source views
        if isinstance(ref_ids, int):
            ref_ids = [ref_ids]
        for ref_id in ref_ids:
            assert ref_id < len(views) and ref_id >= 0
        src_ids = [i for i in range(len(views)) if i not in ref_ids]            

        # #feat: B x S x D  pos: B x S x 2
        with MyNvtxRange('encode'):
            shapes, enc_feats, poses = self._encode_multiview(views)
            pes = self.get_pe(views, ref_ids=ref_ids)
        
        # select and stacck up ref and src elements
        ref_feats, src_feats = self.split_stack_ref_src(enc_feats, ref_ids, src_ids) # (R, B, S, D), (V-R, B, S, D)
        ref_poses, src_poses = self.split_stack_ref_src(poses, ref_ids, src_ids)  # (R, B, S, 2), (V-R, B, S, 2)
        ref_pes, src_pes = self.split_stack_ref_src(pes, ref_ids, src_ids) # (R, B, S, D), (V-R, B, S, D)
        ref_shapes, src_shapes = self.split_stack_ref_src(shapes, ref_ids, src_ids) # (R, B, 2), (V-R, B, 2)
        
        # combine all ref images into object-centric representation
        with MyNvtxRange('decode'):
            dec_feats_ref, dec_feats_src = self._decode_multiview(ref_feats, src_feats, 
                                                                  ref_poses, src_poses, 
                                                                  ref_pes, src_pes)
        
        with MyNvtxRange('head'):
            with torch.cuda.amp.autocast(enabled=False):
                # conf: ((V-R)*B, H, W)  pts3d: ((V-R)*B, H, W, 3)
                res_ref = self._downstream_head(1, [tok.float() for tok in dec_feats_ref], ref_shapes)
                res_src = self._downstream_head(2, [tok.float() for tok in dec_feats_src], src_shapes)
                # print(res_ref['pts3d'].shape, res_src['pts3d'].shape, res_ref['conf'].shape, res_src['conf'].shape)
        
        # split the results back to each view
        results = [] 
        B = res_ref['pts3d'].shape[0] // len(ref_ids)
        for id in range(len(views)):
            res = {}
            if id in ref_ids:
                rel_id = ref_ids.index(id)
                res['pts3d'] = res_ref['pts3d'][rel_id*B:(rel_id+1)*B]
                res['conf'] = res_ref['conf'][rel_id*B:(rel_id+1)*B]
            else:
                rel_id = src_ids.index(id)
                res['pts3d_in_other_view'] = res_src['pts3d'][rel_id*B:(rel_id+1)*B]
                res['conf'] = res_src['conf'][rel_id*B:(rel_id+1)*B]
            results.append(res)
        return results
