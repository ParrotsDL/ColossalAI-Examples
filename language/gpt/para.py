import torch
from titans.model.gpt import gpt2_4B, gpt2_12B, gpt2_13B, gpt2_small
from torchsummary import summary

nums = [0]
def hook(layer_name):
    def true_hook(module, tops, bottoms):
        if isinstance(bottoms, (list, tuple)):
            for x in bottoms:
                if x is not None:
                    nums[0] += x.numel()
            # print(layer_name, [x.shape for x in bottoms if x is not None])
        else:
            nums[0] += bottoms.numel()
            # print(layer_name, bottoms.shape)
    return true_hook


x = torch.rand([2, 1024], dtype=torch.float32)#.cuda()
mm = gpt2_4B()
# mm = gpt2_small()

for layer_name, module in mm.named_modules():
    module.register_forward_hook(hook(layer_name))
print(mm)
# summary(mm, input_size=(1024,))

out = mm(x)

print("End!")
# print("Num of tersors: ",nums)
# print(out.shape)


# mm = gpt2_12B()
# mm = gpt2_13B()
total_params = sum(p.numel() for p in mm.parameters())
print(f'Total parameters: {total_params*1e-9:.5f} B')
print(f'Total element of tensors: {nums[0]*1e-9:.5f} B')
print(f'Total : {(nums[0]+total_params)*1e-9:.5f} B')