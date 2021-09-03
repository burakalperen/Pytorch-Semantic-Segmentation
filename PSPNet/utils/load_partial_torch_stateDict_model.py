import torch
from network.PSPNet import PSPNet
n_classes = 3
use_aux = True

# basically load model with state dict in Pytorch
# model.load_state_dict(torch.load("./checkpoints/deneme_aux_multiclass.pth"))




model = PSPNet(layers=50,num_classes = n_classes,training=False,pretrained=False,use_aux=use_aux)
state_dict = torch.load("./checkpoints/deneme_aux_multiclass.pth")


# to show key and weights
# for k,v in state_dict.items():
#     print(k)#weights
#     print(v)#keys

update_dict = state_dict.copy()

#delete keys that consist of "aux" 
for k in state_dict:
    if "aux" in k:
        del update_dict[k]

for k,v in update_dict.items():
    print(k)#weights
    print(v)#keys

# load new state_dict with strict flag. With strict=False flag, load state_dict function ignore non-matching keys(state dict keys and weights) 
model.load_state_dict(update_dict,strict=False)

print(model.load_state_dict(update_dict,strict=False).missing_keys)


# #test
# for k in model.state_dict():
#     print(k)


