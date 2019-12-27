import torch, sys
import numpy as np
sys.path.append('../')
from utils.args import parse_args

def weight_sensitivity_analysis(model, test_points, std=0.01, num_repeats=20):
    num_params = len(list(model.parameters()))
    with torch.no_grad():
        orig_out = model(test_points)

    layer_outputs = []
    for layer in model.parameters():
        curr_layer_outputs = []
        for i in range(num_repeats):
            gaussian_noise = torch.normal(mean=torch.zeros_like(layer),
                                          std=std)

            with torch.no_grad():
                layer += gaussian_noise
                curr_output = model(test_points)
                layer -= gaussian_noise

                # import pdb; pdb.set_trace()
                abs_diff = torch.abs(orig_out - curr_output)
                mean_abs_diff = torch.mean(abs_diff).data
                curr_layer_outputs.append(mean_abs_diff)
        # print(curr_layer_outputs)
        layer_outputs.append(np.mean(curr_layer_outputs))
    print(np.array(layer_outputs)/std)
    return layer_outputs

if __name__ == '__main__':
    args, device = parse_args()
    model = args.model(hidden_size=4)
    test_points = torch.tensor([1.,2.,3.]).reshape(-1,1)
    import pdb; pdb.set_trace()
    for _ in range(10):
        weight_sensitivity_analysis(model, test_points)




