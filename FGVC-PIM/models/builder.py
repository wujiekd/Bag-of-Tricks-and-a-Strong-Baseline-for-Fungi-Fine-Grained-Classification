import torch
from typing import Union
from torchvision.models.feature_extraction import get_graph_node_names
# from timm.models import create_model
from .pim_module import pim_module

"""
[Default Return]
Set return_nodes to None, you can use default return type, all of the model in this script 
return four layers features.

[Model Configuration]
if you are not using FPN module but using Selector and Combiner, you need to give Combiner a 
projection  dimension ('proj_size' of GCNCombiner in pim_module.py), because graph convolution
layer need the input features dimension be the same.

[Combiner]
You must use selector so you can use combiner.

[About Costom Model]
This function is to building swin transformer. timm swin-transformer + torch.fx.proxy.Proxy 
could cause error, so we set return_nodes to None and change swin-transformer model script to
return features directly.
Please check 'timm/models/swin_transformer.py' line 541 to see how to change model if your costom
model also fail at create_feature_extractor or get_graph_node_names step.
"""

def load_model_weights(model, model_path):
    ### reference https://github.com/TACJu/TransFG
    ### thanks a lot.
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model


def build_resnet50(pretrained: str = "./resnet50_miil_21k.pth",
                   return_nodes: Union[dict, None] = None,
                   num_selects: Union[dict, None] = None, 
                   img_size: int = 448,
                   use_fpn: bool = True,
                   fpn_size: int = 512,
                   proj_type: str = "Conv",
                   upsample_type: str = "Bilinear",
                   use_selection: bool = True,
                   num_classes: int = 200,
                   use_combiner: bool = True,
                   comb_proj_size: Union[int, None] = None):
    
    import timm
    
    if return_nodes is None:
        return_nodes = {
            'layer1.2.act3': 'layer1',
            'layer2.3.act3': 'layer2',
            'layer3.5.act3': 'layer3',
            'layer4.2.act3': 'layer4',
        }
    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }
    
    backbone = timm.create_model('resnet50', pretrained=False, num_classes=11221)
    ### original pretrained path "./models/resnet50_miil_21k.pth"
    if pretrained != "":
        backbone = load_model_weights(backbone, pretrained)

    # print(backbone)
    # print(get_graph_node_names(backbone))
    
    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = return_nodes,
                                   img_size = img_size,
                                   use_fpn = use_fpn,
                                   fpn_size = fpn_size,
                                   proj_type = proj_type,
                                   upsample_type = upsample_type,
                                   use_selection = use_selection,
                                   num_classes = num_classes,
                                   num_selects = num_selects, 
                                   use_combiner = num_selects,
                                   comb_proj_size = comb_proj_size)


def build_efficientnet(pretrained: bool = True,
                       return_nodes: Union[dict, None] = None,
                       num_selects: Union[dict, None] = None, 
                       img_size: int = 448,
                       use_fpn: bool = True,
                       fpn_size: int = 512,
                       proj_type: str = "Conv",
                       upsample_type: str = "Bilinear",
                       use_selection: bool = True,
                       num_classes: int = 200,
                       use_combiner: bool = True,
                       comb_proj_size: Union[int, None] = None):

    import torchvision.models as models

    if return_nodes is None:
        return_nodes = {
            'features.4': 'layer1',
            'features.5': 'layer2',
            'features.6': 'layer3',
            'features.7': 'layer4',
        }
    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }
    import timm
    # backbone = models.efficientnet_b7(pretrained=pretrained)
    backbone = timm.create_model('tf_efficientnet_b7_ns', pretrained=False)
    # model = create_model(
    #     args.model,
    #     pretrained=args.pretrained,
    #     num_classes=args.num_classes,
    #     drop_rate=args.drop,
    #     drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
    #     drop_path_rate=args.drop_path,
    #     drop_block_rate=args.drop_block,
    #     global_pool=args.gp,
    #     bn_momentum=args.bn_momentum,
    #     bn_eps=args.bn_eps,
    #     scriptable=args.torchscript,
    #     checkpoint_path=args.initial_checkpoint)

    checkpoint = torch.load('/home/data4/czp/FGVC2022_Snake/WorkspaceRecord/tf_efficient_imgsize600_freeze_2_resume/20220413-174356-tf_efficientnet_b7_ns-600/Best_f1_score.pth.tar', map_location=torch.device('cpu'))
    if 'state_dict' in checkpoint:
        state_dict_key = 'state_dict'
        state_dict = checkpoint[state_dict_key]
        state_dict['classifier.weight']=torch.randn([1000,2560])
        state_dict['classifier.bias']=torch.randn(1000)
    elif 'model' in checkpoint:
        state_dict_key = 'model'
        state_dict = checkpoint[state_dict_key]
        state_dict['classifier.weight']=torch.randn([1000,2560])
        state_dict['classifier.bias']=torch.randn(1000)
    backbone.load_state_dict(state_dict,strict=False)

#   freeze layer
    for block in list(backbone.children())[:3]:
            for param in list(block.parameters()):
                param.requires_grad = False
    for block in list(backbone.children())[3][:2]:
            for param in list(block.parameters()):
                param.requires_grad = False
    backbone.train()

    # print(backbone)
    # print(get_graph_node_names(backbone))
    ## features.1~features.7

    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = return_nodes,
                                   img_size = img_size,
                                   use_fpn = use_fpn,
                                   fpn_size = fpn_size,
                                   proj_type = proj_type,
                                   upsample_type = upsample_type,
                                   use_selection = use_selection,
                                   num_classes = num_classes,
                                   num_selects = num_selects, 
                                   use_combiner = num_selects,
                                   comb_proj_size = comb_proj_size)




def build_vit16(pretrained: str = "./vit_base_patch16_224_miil_21k.pth",
                return_nodes: Union[dict, None] = None,
                num_selects: Union[dict, None] = None, 
                img_size: int = 448,
                use_fpn: bool = True,
                fpn_size: int = 512,
                proj_type: str = "Linear",
                upsample_type: str = "Conv",
                use_selection: bool = True,
                num_classes: int = 200,
                use_combiner: bool = True,
                comb_proj_size: Union[int, None] = None):

    import timm
    
    backbone = timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=False)
    ### original pretrained path "./models/vit_base_patch16_224_miil_21k.pth"
    if pretrained != "":
        backbone = load_model_weights(backbone, pretrained)

    backbone.train()

    # print(backbone)
    # print(get_graph_node_names(backbone))
    # 0~11 under blocks

    if return_nodes is None:
        return_nodes = {
            'blocks.8': 'layer1',
            'blocks.9': 'layer2',
            'blocks.10': 'layer3',
            'blocks.11': 'layer4',
        }
    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }

    ### Vit model input can transform 224 to another, we use linear
    ### thanks: https://github.com/TACJu/TransFG/blob/master/models/modeling.py
    import math
    from scipy import ndimage

    posemb_tok, posemb_grid = backbone.pos_embed[:, :1], backbone.pos_embed[0, 1:]
    posemb_grid = posemb_grid.detach().numpy()
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = img_size//16
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    backbone.pos_embed = torch.nn.Parameter(posemb)

    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = return_nodes,
                                   img_size = img_size,
                                   use_fpn = use_fpn,
                                   fpn_size = fpn_size,
                                   proj_type = proj_type,
                                   upsample_type = upsample_type,
                                   use_selection = use_selection,
                                   num_classes = num_classes,
                                   num_selects = num_selects, 
                                   use_combiner = num_selects,
                                   comb_proj_size = comb_proj_size)


def build_swintransformer(pretrained: bool = True,
                          num_selects: Union[dict, None] = None, 
                          img_size: int = 384,
                          use_fpn: bool = True,
                          fpn_size: int = 512,
                          proj_type: str = "Linear",
                          upsample_type: str = "Conv",
                          use_selection: bool = True,
                          num_classes: int = 200,
                          use_combiner: bool = True,
                          comb_proj_size: Union[int, None] = None):
    """
    This function is to building swin transformer. timm swin-transformer + torch.fx.proxy.Proxy 
    could cause error, so we set return_nodes to None and change swin-transformer model script to
    return features directly.
    Please check 'timm/models/swin_transformer.py' line 541 to see how to change model if your costom
    model also fail at create_feature_extractor or get_graph_node_names step.
    """

    import timm

    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }

    backbone = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=pretrained)

    checkpoint = torch.load('/home/data1/lkd/CVPR_FUNGI/Fungi_data/Fungidata/linshi_model/swin_large_extra2_data1_largelr/checkpoint-4.pth.tar', map_location=torch.device('cpu'))
    
    if 'state_dict' in checkpoint:
        state_dict_key = 'state_dict'
        state_dict = checkpoint[state_dict_key]
        state_dict['head1.weight']=state_dict.pop('head.weight')
        state_dict['head1.bias']=state_dict.pop('head.bias')
    elif 'model' in checkpoint:
        state_dict_key = 'model'
        state_dict = checkpoint[state_dict_key]
        state_dict['head1.weight']=state_dict.pop('head.weight')
        state_dict['head1.bias']=state_dict.pop('head.bias')
    backbone.load_state_dict(state_dict,strict=False)

    print(backbone)
    # print(get_graph_node_names(backbone))
    backbone.train()
    
    print("Building...")
    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = None,
                                   img_size = img_size,
                                   use_fpn = use_fpn,
                                   fpn_size = fpn_size,
                                   proj_type = proj_type,
                                   upsample_type = upsample_type,
                                   use_selection = use_selection,
                                   num_classes = num_classes,
                                   num_selects = num_selects, 
                                   use_combiner = num_selects,
                                   comb_proj_size = comb_proj_size)


def build_swintransformer_base(pretrained: bool = True,
                          num_selects: Union[dict, None] = None, 
                          img_size: int = 384,
                          use_fpn: bool = True,
                          fpn_size: int = 512,
                          proj_type: str = "Linear",
                          upsample_type: str = "Conv",
                          use_selection: bool = True,
                          num_classes: int = 200,
                          use_combiner: bool = True,
                          comb_proj_size: Union[int, None] = None):
    """
    This function is to building swin transformer. timm swin-transformer + torch.fx.proxy.Proxy 
    could cause error, so we set return_nodes to None and change swin-transformer model script to
    return features directly.
    Please check 'timm/models/swin_transformer.py' line 541 to see how to change model if your costom
    model also fail at create_feature_extractor or get_graph_node_names step.
    """

    import timm

    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }

    backbone = timm.create_model('swin_base_patch4_window12_384', pretrained=pretrained)

    checkpoint = torch.load('/home/data1/lkd/CVPR_FUNGI/Fungi_data/Fungidata/linshi_model/swin_base_extra2_data1_largelr/checkpoint-4.pth.tar', map_location=torch.device('cpu'))
    
    if 'state_dict' in checkpoint:
        state_dict_key = 'state_dict'
        state_dict = checkpoint[state_dict_key]
        state_dict['head1.weight']=state_dict.pop('head.weight')
        state_dict['head1.bias']=state_dict.pop('head.bias')
    elif 'model' in checkpoint:
        state_dict_key = 'model'
        state_dict = checkpoint[state_dict_key]
        state_dict['head1.weight']=state_dict.pop('head.weight')
        state_dict['head1.bias']=state_dict.pop('head.bias')
    backbone.load_state_dict(state_dict,strict=False)

    print(backbone)

    # print(get_graph_node_names(backbone))
    backbone.train()
    
    print("Building...")
    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = None,
                                   img_size = img_size,
                                   use_fpn = use_fpn,
                                   fpn_size = fpn_size,
                                   proj_type = proj_type,
                                   upsample_type = upsample_type,
                                   use_selection = use_selection,
                                   num_classes = num_classes,
                                   num_selects = num_selects, 
                                   use_combiner = num_selects,
                                   comb_proj_size = comb_proj_size)

if __name__ == "__main__":
    ### ==== resnet50 ====
    # model = build_resnet50(pretrained='./resnet50_miil_21k.pth')
    # t = torch.randn(1, 3, 448, 448)
    
    ### ==== swin-t ====
    model = build_swintransformer(False)
    t = torch.randn(1, 3, 384, 384)

    ### ==== vit ====
    # model = build_vit16(pretrained='./vit_base_patch16_224_miil_21k.pth')
    # t = torch.randn(1, 3, 448, 448)

    # ### ==== efficientNet ====
    # model = build_efficientnet(pretrained=False)
    # t = torch.randn(1, 3, 448, 448)

    model.cuda()
    
    t = t.cuda()
    outs = model(t)
    for out in outs:
        print(type(out))
        print("    " , end="")
        if type(out) == dict:
            print([name for name in out])


MODEL_GETTER = {
    "resnet50":build_resnet50,
    "swin-t":build_swintransformer,
    "vit":build_vit16,
    "efficient":build_efficientnet,
    "swint-base":build_swintransformer_base
}
