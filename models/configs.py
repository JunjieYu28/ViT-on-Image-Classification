# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_new():
    """Returns our configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)})
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.01
    config.transformer.dropout_rate = 0.01
    config.transformer.prob_pass = 0.01
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_test():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 288
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 768
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.01
    config.transformer.dropout_rate = 0.1
    config.transformer.prob_pass = 0.01
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_set_288_288():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 288
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 288
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_set_288_384():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 288
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_set_288_768():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 288
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 768
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_set_384_768():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 768
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_nh_8():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 8
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_nh_16():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 16
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_nb_4():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_nb_12():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_ps_2():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (2, 2), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 0 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_ps_4():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 0 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_ps_8():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 0 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_res_0():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 0 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_res_1():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 1 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_res_2():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_adp_0():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_adp_1Eng1():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_adp_1Eng2():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.01
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_adp_1Eng3():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.001
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_dp_0():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_dp_1Eng1():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_dp_1Eng2():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.01
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_dp_1Eng3():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.001
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_pp_0():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.0
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_pp_1Eng1():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_pp_1Eng2():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.01
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_pp_1Eng3():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_final_pp_1Eng4():
    """Returns our final configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4), 'grid_1': (4, 4), 'grid_2': (8, 8)})
    config.ResNet_type = 2 # 0 means not setting resnet
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.0001
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config_Stochastic_Depth():
    """Returns our configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)})
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384
    config.transformer.num_heads = 12
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.prob_pass = 0.01
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_config():
    """Returns our configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_our_ResNet_plus_config():
    """Returns the ResNet + ViT configuration."""
    config = get_our_config_new()
    del config.patches.size
    config.patches.grid = (8, 8)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    return config

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    del config.patches.size
    config.patches.grid = (14, 14)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config
