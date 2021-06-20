import cv2 as cv
import torch
# import stylegan2
import legacy
import dnnlib
import copy
import math

def extract_conv_names(model):
    # layers are synthesis.b{res}/...
    # make a list of (name, resolution, level, position)
    # Currently assuming square(?)

    model_names = list(model.state_dict().keys())
    conv_names = []

    resolutions =  [4*2**x for x in range(9)]
    level_names = [["conv0", "const"],
                    ["conv1", "torgb"]]
    
    position = 0
    # option not to split levels
    for res in resolutions:
        root_name = f"synthesis.b{res}."
        for level, level_suffixes in enumerate(level_names):
            for suffix in level_suffixes:
                search_name = root_name + suffix
                matched_names = [x for x in model_names if x.startswith(search_name)]
                to_add = [(name, res, level, position) for name in matched_names]
                conv_names.extend(to_add)
            position += 1

    return conv_names


def blend_models(model_1, model_2, resolution, level, blend_width=None, verbose=False):
    resolutions = [4 * 2 ** i for i in range(8)]
    mid = resolutions.index(resolution)



    model_1_state_dict = model_1.state_dict()
    model_2_state_dict = model_2.state_dict()
    assert(model_1_state_dict.keys() == model_2_state_dict.keys())
    
    model_out = copy.deepcopy(model_1)

    model_1_names = extract_conv_names(model_1)
    model_2_names = extract_conv_names(model_2)

    
    short_names = [(x[1:3]) for x in model_1_names]
    full_names = [(x[0]) for x in model_1_names]
    mid_point_idx = short_names.index((resolution, level))
    mid_point_pos = model_1_names[mid_point_idx][3]
    
    ys = []
    done_res = 0
    for name, res, level, position in model_1_names:
        # low to high (res)
        x = position - mid_point_pos
        
        if blend_width:
            exponent = -x/blend_width
            y = 1 / (1 + math.exp(exponent))
        else:
            y = 1 if x > 1 else 0

        if y < 0.55:
          y = 0

        # print(f'resolution {resolution}, layer {res}, x, {x}, width {blend_width}, y {y}')

        ys.append(y)
        
        if verbose:
            # if done_res != res:
              # print(f"width {blend_width}, Blending {res} by {y}")
            done_res = res
    
    out_state = model_out.state_dict()
    for y, (layer, _, _, _) in zip(ys, model_1_names):
        out_state[layer] = y * model_2_state_dict[layer] + \
            (1 - y) * model_1_state_dict[layer]
    model_out.load_state_dict(out_state)
    return model_out
    
    
    
    tfutil.set_vars(
        tfutil.run(
            {model_out.vars[name]: (model_2.vars[name] * y + model_1.vars[name] * (1-y))
             for name, y 
             in zip(full_names, ys)}
        )
    )

    return model_out

    layers = []
    ys = []
    for k, v in model_1_state_dict.items():
        import pdb
        pdb.set_trace()
        if k.startswith('G_synthesis.conv_blocks.'):
            pos = int(k[len('G_synthesis.conv_blocks.')])
            x = pos - mid
            if blend_width:
                exponent = -x / blend_width
                y = 1 / (1 + math.exp(exponent))
            else:
                y = 1 if x > 0 else 0

            layers.append(k)
            ys.append(y)
        elif k.startswith('G_synthesis.to_data_layers.'):
            pos = int(k[len('G_synthesis.to_data_layers.')])
            x = pos - mid
            if blend_width:
                exponent = -x / blend_width
                y = 1 / (1 + math.exp(exponent))
            else:
                y = 1 if x > 0 else 0
            layers.append(k)
            ys.append(y)
    out_state = G_out.state_dict()
    for y, layer in zip(ys, layers):
        out_state[layer] = y * model_2_state_dict[layer] + \
            (1 - y) * model_1_state_dict[layer]
    G_out.load_state_dict(out_state)
    return G_out


# def main():
#     G_out = blend_models("checkpoints/stylegan2_512x512_with_pretrain/pretrain/Gs.pth",
#                          "checkpoints/stylegan2_512x512_with_pretrain_new_2/20000_2020-12-23_13-17-51/Gs.pth",
#                          8,
#                          None)
#     G_out.save('G_blend.pth')


# if __name__ == '__main__':
#     main()
