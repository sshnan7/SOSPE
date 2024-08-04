import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .convnext_ver2310 import build_convnext
#from .convnext_ver2309 import build_convnext
from .convnext3d_ver2312 import build_convnext

from random import randint
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncodingPermute2D 
from timm.models.layers import trunc_normal_

# helpers
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
            
class DataDecoder(nn.Module):
    def __init__(self, emb_dim, kernel_size = 3, groups = 4, depth = 3):
        super().__init__()

        def make_block(emb_dim_v, kernel_size_v, groups_v):
            block = [
                nn.Conv2d(emb_dim_v, emb_dim_v, kernel_size_v, padding=kernel_size_v // 2, groups=groups_v),
                LayerNorm(emb_dim_v, eps=1e-6, data_format="channels_first"),
                nn.GELU()
            ]
            return nn.Sequential(*block)

        self.blocks = nn.Sequential(
            *[
                make_block(emb_dim, kernel_size, groups)
                for i in range(depth)
            ]
        )

        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x = rearrange(x,'b token_len emb -> b emb token_len')
        x = x.unsqueeze(2)

        for i, layer in enumerate(self.blocks):
            ident = x.clone()
            x = layer(x)
            x = x + ident
        
        x = x.squeeze(2)
        x = rearrange(x,'b emb token_len -> b token_len emb')
        x = self.proj(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, argument, batch, in_chans, kernel, emb_dim, depth, block_stride, tokenizer = None, dropout = 0., group_idx = 4, rx_num_masking = 0, tx_num_masking = 0, pretrain = False, mask_ratio = 0):
        super().__init__()
        
        in_chan = in_chans
        #blocks
        self.indices = self.sliding_window(windowsize = int((in_chans)**(1/2)), stride = block_stride)
        self.batch_size = batch
        
        self.in_chan = in_chans
        self.rx_num_masking = rx_num_masking
        self.tx_num_masking = tx_num_masking
        self.pretrain = pretrain
        self.group_idx = group_idx
        self.mask_ratio = mask_ratio
        self.kernel = kernel
        self.inv_block_size = argument.inv_mask_block
        self.original_data2vec = argument.original_data2vec
        signal_len = 768
        emb_dim = emb_dim
        heads = emb_dim //32
        
        self.emb_type = "convnext3d"
        print("emb type", self.emb_type)
        
        #if in_chans != 1 :
        if self.emb_type == "convnext" or "convnext3d" :
            self.patch_emb_net = tokenizer
        
        if self.emb_type == "linear" :
            self.patch_emb_net = nn.Sequential(
                nn.Linear(768, (768+128)//2, bias=False),
                nn.LayerNorm((768+128)//2),
                nn.GELU(),
                nn.Linear((768+128)//2, (768+128)//2, bias=False),
                nn.LayerNorm((768+128)//2),
                nn.GELU(),
                nn.Linear((768+128)//2, 128, bias=False),
                #nn.Linear(768, 128, bias=False),
            )
        if self.emb_type == "transformer" :
            encoder_layer1 = nn.TransformerEncoderLayer(d_model=768, nhead=16, dim_feedforward=768*4, batch_first=True, activation="gelu")
            self.patch_emb_net = nn.Sequential(
                nn.Linear(768, 768, bias=False),
                nn.TransformerEncoder(encoder_layer1, num_layers=6),
                nn.Linear(768, 128, bias=False),
            )
        
        '''
        if in_chans == 1 :
            self.to_patch_embedding = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, stride= 3),
                LayerNorm(16, eps=1e-6, data_format="channels_first"),
            )
            self.emb_type = ["light emb"]
        '''
        self.dropout = nn.Dropout(0.4)
                
        self.layernorm1 = nn.LayerNorm(256, eps=1e-06)
        self.mask_emb = nn.Parameter(torch.Tensor(1, 1, emb_dim))
        nn.init.trunc_normal_(self.mask_emb, 0.02)
        self.pos_enc = PositionalEncoding1D(emb_dim)
        
        self.transformer_encoder = nn.ModuleList([])
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=argument.nheads, dim_feedforward=emb_dim*4, batch_first=True, activation="gelu")
        #self.transformer_init_weights(encoder_layer)
        for i in range(depth) :
            self.transformer_encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=1))
        
        if argument.pretrain or argument.downstream or argument.fineresume or argument.pretrain_model :
            if self.original_data2vec :
                self.pretrain_crossattn = nn.ModuleList([])
                encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=argument.nheads, dim_feedforward=emb_dim*4, batch_first=True, activation="gelu")
                for i in range(argument.dec_layers) : #cross-self decoder
                    self.pretrain_crossattn.append(nn.TransformerEncoder(encoder_layer, num_layers=1))
            else :
                self.pretrain_crossattn = nn.ModuleList([])
                decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=argument.nheads, dim_feedforward=emb_dim*4, batch_first=True, activation="gelu")
                for i in range(argument.dec_layers) : #cross-self decoder
                    self.pretrain_crossattn.append(nn.TransformerDecoder(decoder_layer, num_layers=1))
                    
            encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=argument.nheads, dim_feedforward=emb_dim*4, batch_first=True, activation="gelu")
            self.pretrain_crossattn.append(nn.TransformerEncoder(encoder_layer, num_layers=1)) #last layer selfattn
            
            self.decoder_proj_raw = nn.Linear(emb_dim, kernel*16)
            self.decoder_proj_vec = nn.Linear(emb_dim, emb_dim)
        
        #self.decoder = DataDecoder(emb_dim)
        #self.regression_head = nn.Linear(emb_dim, emb_dim)
        
        self.apply(self._init_weights)
        
        dummy = torch.zeros((self.batch_size, 64, 768))
        self.forward(dummy, init=True)
    
    def transformer_init_weights(self, layer):
        if hasattr(layer, 'weight'):
            trunc_normal_(layer.weight, std=.02)
            #print(layer)
            #torch.nn.init.xavier_uniform_(layer.weight)
            #nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias != None :
                nn.init.constant_(m.bias, 0)
        #if isinstance(m, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
        #    trunc_normal_(m.weight, std=.02)
        #    if m.bias != None :
        #        nn.init.constant_(m.bias, 0)
    
    def print_func(self, string1, string2, init) :
        if init :
            print(string1, string2)
    
    def sliding_window(self, windowsize, stride) :
        a = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
              [8, 9, 10, 11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20, 21, 22, 23],
              [24, 25, 26, 27, 28, 29, 30, 31],
              [32, 33, 34, 35, 36, 37, 38, 39],
              [40, 41, 42, 43, 44, 45, 46, 47],
              [48, 49, 50, 51, 52, 53, 54, 55],
              [56, 57, 58, 59, 60, 61, 62, 63]]).detach()
        #initial
        indices = a[:windowsize, :windowsize].unsqueeze(0)
        indices = rearrange(indices, 'b y x -> b (y x)')
        for i in range(8-(windowsize-1)) :
            for j in range(8-(windowsize-1)) :
                if i==0 and j==0 :
                    continue
                if i%stride != 0 or j%stride != 0 :
                    continue
                else :
                    tmp_indices = a[i:i+windowsize, j:j+windowsize].unsqueeze(0)
                    tmp_indices = rearrange(tmp_indices, 'b y x -> b (y x)')
                    indices = torch.cat((indices, tmp_indices), 0)
        return indices
    
    def pickup_group(self, x) :
        batch = x.shape[0]
        
        if self.emb_type ==  "convnext3d" :
            input_tensor = rearrange(x, 'b (Rx Tx) ch len -> b ch Rx Tx len', b = batch, Rx = 8, Tx = 8)
            
            if self.pretrain :
                input_tensor_cropped = input_tensor[:,:,:4, :4, :] #group 0
            
                #for target
                if self.group_idx <= 3 :
                    input_tensor = rearrange(input_tensor, 'b ch (Rx1 Rx2) (Tx1 Tx2) len -> b ch (Rx1 Tx1) Rx2 Tx2 len', b = batch, Rx2 = 4, Tx2 = 4)
                    input_tensor = input_tensor[:, :, self.group_idx, :, :, :]
                    input_tensor.squeeze(2)
                    
                return input_tensor, input_tensor_cropped
                
            else :
                if self.group_idx <=3 :
                    
                    input_tensor = rearrange(input_tensor, 'b ch (Rx1 Rx2) (Tx1 Tx2) len -> b ch (Rx1 Tx1) Rx2 Tx2 len', b = batch, Rx2 = 4, Tx2 = 4)
                    input_tensor = input_tensor[:, :, self.group_idx, :, :, :]
                    input_tensor.squeeze(2)
                
                return input_tensor
            
        else :
            if selfsupervised :
                block_num = len(self.indices)
                self.randomint = randint(0, block_num-1)
                #indices_backup = self.indices
                pick_up_indices = self.indices[self.randomint].unsqueeze(0)
                input_tensor = torch.index_select(x, 1, pick_up_indices[0])
            else :
                input_tensor = torch.index_select(x, 1, self.indices[0])
                for i in range(1, len(self.indices)) :
                    tmp_select = torch.index_select(x, 1, self.indices[i])
                    input_tensor = torch.cat((input_tensor, tmp_select), 0)
                
                input_tensor = rearrange(input_tensor, '(block b) ch st len -> (b block) ch st len', b = batch)
        
        return input_tensor
    
    #def mask_block(self, x, rx_num_masking = 0, tx_num_masking = 0, ratio) :
    def masking(self, x, num_group = 4, block_size=1) :
        batch = x.shape[0]
        patch_len = 768//self.kernel
        block_len = patch_len//block_size
        len_keep = int(patch_len * (1 - self.mask_ratio))
        block_keep = len_keep//block_size
        
        noise = torch.rand(batch, num_group, block_len, device=x.device) #batch group kernel_len
        if block_size != 1 :
            noise = noise.unsqueeze(-1).repeat(1, 1, 1, block_size)
            noise = rearrange(noise, 'b group len block -> b group (len block)')
        
        # sort noise for each sample      
        ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep until len_keep, over len_keep is remove
        ids_restore = torch.argsort(ids_shuffle, dim=2)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :, :len_keep]
        
        #x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch, num_group, patch_len], device=x.device)
        mask[:, :, :len_keep] = 0
        
        # unshuffle to get the binary mask
        emb_mask = torch.gather(mask, dim=2, index=ids_restore)
        
        return emb_mask.detach(), ids_restore.detach(), ids_keep.detach()
    
    def patch_embedding(self, x) :
        if self.emb_type != "convnext" and self.emb_type != "convnext3d" :
            x = x.squeeze(1)
            x = rearrange(x, '(b block) len emb -> b (block len) emb', b = b)   
        if self.emb_type == "convnext" :
            encoder_output = rearrange(encoder_output_patches, '(b block) len emb -> b (block len) emb', b = b)
            
        encoder_output = self.patch_emb_net(x)
        
        return encoder_output
    
    def forward_selfattn(self, x, ids_keep= None, patch_len = 1, group_pos = None, group_restore = None) : #ids_keep : what number patch to keep
        device = x.device
        b, patch_num, emb = x.shape
        dummy_patches = torch.zeros((b, patch_len, emb)).to(device)
        
        if ids_keep != None : #masking
            ids_b, group, ids_patch_n = ids_keep.shape
            
            pos_enc = self.pos_enc(dummy_patches).to(device)
            x += pos_enc.detach()
            x = rearrange(x, 'b (group len) emb -> b group len emb', group = 4) 
            
            if group_restore != None : # pick up retain group
                x = x[:,group_restore,:,:]
            
            #masking
            x = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, emb))
            x = rearrange(x, 'b group len emb -> b (group len) emb') #for transformer input
            
        else :#not masking
            pos_enc = self.pos_enc(dummy_patches).to(device)
            if group_pos !=None :
                pos_enc = pos_enc[:, patch_num*group_pos:patch_num*(group_pos+1), :] 
            x += pos_enc.detach()
            
            
        x = self.transformer_encoder[0](x)
        for i in range(1, len(self.transformer_encoder)) :
            x = self.transformer_encoder[i](x)
        
        
        return x
    
    def forward_crossattn(self, x, y = None, groups_pos = None) :
        device = x.device
        if groups_pos != None : # means x have 3 groups
            b, groups, group_patch_len, emb = x.shape
            patch_len = group_patch_len*4
            dummy_patches = torch.zeros((b, group_patch_len*4, emb)).to(device)
            x_pos_enc = self.pos_enc(dummy_patches).detach().to(device)
            x_pos_enc = rearrange(x_pos_enc, 'b (group len) emb -> b group len emb', group = 4) 
            x_pos_enc = x_pos_enc[:,groups_pos,:,:]
            x_pos_enc = rearrange(x_pos_enc, 'b group len emb -> b (group len) emb')
            x = rearrange(x, 'b group len emb -> b (group len) emb')
        else :
            x_pos_enc = self.pos_enc(x).detach().to(device)
        x = x.to(device)
        x += x_pos_enc
        
        if y != None :
            y_pos_enc = self.pos_enc(y).detach().to(device)
            memory = y.clone()
            memory = memory + y_pos_enc
        for i in range(len(self.pretrain_crossattn)-1) :
            if y != None :
                x = self.pretrain_crossattn[i](x, memory) #cross attn
            else :
                x = self.pretrain_crossattn[i](x) #only name is crossattn // real selfattn
        
        x = self.pretrain_crossattn[-1](x) # self attn final layer
            
        #x = self.pretrain_selfattn[0](x)
        #for i in range(1, len(self.pretrain_selfattn)) :
        #    x = self.pretrain_selfattn[i](x)
        raw = self.decoder_proj_raw(x)
        vec = self.decoder_proj_vec(x)
        
        return raw, vec

    
    def task_forward(self, input_tensor, init = False) :
        self.print_func("task", "mode", init)
        x = input_tensor.clone()
        #if self.tx_num_masking > 0 or self.rx_num_masking > 0 :
        #    x = self.mask_block(x) 
        if self.mask_ratio > 0 :
            emb_mask, ids_restore, ids_keep = self.masking(x)
        
        encoder_output_patches = self.patch_embedding(x)
        
        self.print_func("encoder output patches shape ", encoder_output_patches.shape, init)
        self.print_func("transformer number layers ", len(self.transformer_encoder), init)
        
        if len(self.transformer_encoder) > 0 :
            if self.group_idx <= 3 :
                encoder_output_patches = self.forward_selfattn(encoder_output_patches, patch_len = encoder_output_patches.shape[1]*4, group_pos = self.group_idx)
            else :
                encoder_output_patches = self.forward_selfattn(encoder_output_patches, patch_len = encoder_output_patches.shape[1])
        
        self.print_func("backbone finish", encoder_output_patches.shape, init)
        
        return encoder_output_patches
    
    
    def pretrain_forward(self, input_tensor, init = False) :
        self.print_func("pretrain", "mode", init)
        x = input_tensor.clone()
        b = x.shape[0]
        device = x.device        
        
        random_group_position = int(torch.randint(0, 4, (1,)).detach()) #will be memory group
        if self.original_data2vec :
            random_group_position = 5 #False masking
        retain_idx = []
        for i in range(4) :
            if i != random_group_position :
                retain_idx.append(i)
        
        #raw target
        target_raw = input_tensor.clone()
        target_raw = rearrange(target_raw, 'b ch (Rx1 Rx2) (Tx1 Tx2) len -> b (ch Rx1 Tx1) Rx2 Tx2 len', Rx1 = 2, Tx1 = 2)
        target_raw = target_raw[:,retain_idx,:,:]
        self.print_func("pretrain target raw shape ", target_raw.shape, init)
            
        #vec target
        pretrain_target = input_tensor.clone()
        with torch.no_grad():
            pretrain_target = self.patch_embedding(pretrain_target)
            target_vec = self.forward_selfattn(pretrain_target)
        group_len = target_vec.shape[1]//4
        target_vec = rearrange(target_vec, 'b (g g_len) emb -> b g g_len emb', g_len = group_len)
        target_vec = target_vec[:,retain_idx,:, :]
        target_vec = rearrange(target_vec, 'b g g_len emb -> b (g g_len) emb')
        self.print_func("pretrain target vec shape ", target_vec.shape, init)
         
        if self.mask_ratio > 0  :
            if self.original_data2vec :
                emb_mask, ids_restore, ids_keep = self.masking(x, num_group = 4, block_size = self.inv_block_size)
            else :
                emb_mask, ids_restore, ids_keep = self.masking(x, num_group = 3, block_size = self.inv_block_size)
        
        nonmask_backbone_patches = self.patch_embedding(input_tensor)
        nonmask_backbone_patches_grouping = rearrange(nonmask_backbone_patches, 'b (group len) emb -> b group len emb', group = 4)
        
        if not self.original_data2vec :
            memory_backbone_patches = nonmask_backbone_patches_grouping[:,random_group_position,:,:].squeeze(1)
            memory_backbone_patches = self.forward_selfattn(memory_backbone_patches, patch_len = memory_backbone_patches.shape[1]*4, group_pos = random_group_position)
            self.print_func("memory patches shape ", memory_backbone_patches.shape, init)
        
        if self.mask_ratio > 0 :
            mask_encoder_output_patches = self.forward_selfattn(nonmask_backbone_patches, ids_keep, patch_len = nonmask_backbone_patches_grouping.shape[2]*4, group_restore = retain_idx)
        else :
            mask_encoder_output_patches = self.forward_selfattn(mask_backbone_patches) #nonmasking
        self.print_func("kill mask patches shape ", mask_encoder_output_patches.shape, init)
        self.print_func("self attn finish shape ", mask_encoder_output_patches.shape, init)
        
        mask_encoder_output_patches = rearrange(mask_encoder_output_patches, 'b (group len) emb -> b group len emb', group = len(retain_idx))
        self.print_func("encoder output and grouping shape with out masking patch", mask_encoder_output_patches.shape, init)
        
        b, g, patch_num, emb = mask_encoder_output_patches.shape
        mask_tokens = self.mask_emb.repeat(b, g, ids_restore.shape[2] - patch_num, 1)
        mask_encoder_output_patches = torch.cat([mask_encoder_output_patches, mask_tokens], dim=2)  #0 : batch, 1: group, 2 : patch num # no cls token
        mask_encoder_output_patches = torch.gather(mask_encoder_output_patches, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, emb))  # unshuffle
        
        if self.original_data2vec :
            raw_pred, vec_pred = self.forward_crossattn(mask_encoder_output_patches, groups_pos = retain_idx)
        else :
            raw_pred, vec_pred = self.forward_crossattn(mask_encoder_output_patches, memory_backbone_patches, groups_pos = retain_idx)
        
        '''
        raw_pred_list = []
        vec_pred_list = []
        for i in range(len(retain_idx)) :
            tmp_raw_pred, tmp_vec_pred = self.forward_crossattn(mask_encoder_output_patches, memory_backbone_patches, groups_pos = retain_idx)
            raw_pred_list.append(tmp_raw_pred)
            vec_pred_list.append(tmp_vec_pred)
        
        raw_pred = torch.cat((raw_pred_list[0].unsqueeze(1), raw_pred_list[1].unsqueeze(1), raw_pred_list[2].unsqueeze(1)), dim = 1)
        vec_pred = torch.cat((vec_pred_list[0].unsqueeze(1), vec_pred_list[1].unsqueeze(1), vec_pred_list[2].unsqueeze(1)), dim = 1)
        '''
        
        self.print_func("raw pred finish", raw_pred.shape, init)
        self.print_func("vec pred finish", vec_pred.shape, init)
        
        return raw_pred, vec_pred, target_raw, target_vec, emb_mask
        
    
    def forward(self, input_tensor, init=False):
        self.print_func("init input", input_tensor.shape, init)
        self.print_func("input channel", self.in_chan, init)
            
        device = input_tensor.device
        self.indices = self.indices.to(device)
        self.block_num = len(self.indices)
        b, ch, length = input_tensor.shape
        input_tensor = input_tensor.view(b, ch, -1, length)
        
        self.print_func("make 2d sig", input_tensor.shape, init)
        
        x = input_tensor.clone()
        #2d(rxtx len) signal to 3d(rx tx len) signal 
        if self.pretrain :
            input_tensor, input_tensor_cropped = self.pickup_group(x)
        else :
            input_tensor = self.pickup_group(x)
            
        self.print_func("group pickup finish ",input_tensor.shape, init)
        
        #pretrain forward
        if self.pretrain :
            raw_pred, vec_pred, target_raw, target_vec, emb_mask = self.pretrain_forward(input_tensor, init = init)
            
            return raw_pred, vec_pred, target_raw, target_vec, emb_mask
            
        #task forward
        else :
            task_backbone_vec = self.task_forward(input_tensor, init)
            
            return task_backbone_vec

class Self_Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.kernel = args.kernel
        self.criterion = nn.MSELoss()
        self.original_data2vec = args.original_data2vec
        if self.original_data2vec :
            self.group_num = 4
        else :
            self.group_num = 3
        
        self.loss_weight = {'raw' : args.raw_loss_coef, 'vec' : args.vec_loss_coef}
        
    def patchify(self, target_signal, target_vector) :
        target_signal = rearrange(target_signal, 'b group Rx2 Tx2 (len1 len2) -> b group len1 (len2 Rx2 Tx2) ', len2 = self.kernel)

        
        if self.original_data2vec :
            target_vector = rearrange(target_vector, 'b (group len) emb -> b group len emb ', group = self.group_num)
        else :
            target_vector = rearrange(target_vector, 'b (group len) emb -> b group len emb ', group = self.group_num)
        
        return target_signal, target_vector
        
    def forward(self, pred_signal, pred_vec, target_signal, target_vector, emb_mask):
        target_signal, target_vector = self.patchify(target_signal, target_vector)
        #emb_mask = emb_mask[:, 1:, :] #discard group 0 embmask
        pred_signal = rearrange(pred_signal, 'b (group group_patch_len) emb -> b group group_patch_len emb ', group = self.group_num)
        pred_vec = rearrange(pred_vec, 'b (group group_patch_len) emb -> b group group_patch_len emb ', group = self.group_num)
        sig_loss = (pred_signal - target_signal.detach()) ** 2
        sig_loss = sig_loss.mean(dim=-1)  # [N, L], mean loss per patch
        sig_loss = (sig_loss * emb_mask).sum() / emb_mask.sum()  # mean loss on removed patches
        
        vec_loss = (pred_vec - target_vector.detach()) ** 2
        vec_loss = vec_loss.mean(dim=-1)  # [N, L], mean loss per patch
        vec_loss = (vec_loss * emb_mask).sum() / emb_mask.sum()  # mean loss on removed patches
        
        #loss = self.criterion(pred*emb_mask, target*emb_mask)
        losses = {
            'raw' : sig_loss,
            'vec' : vec_loss
        }
        
        loss = self.loss_weight['raw']*losses['raw'] + self.loss_weight['vec']*losses['vec']
        
        return losses, loss
        
'''

if selfsupervised :
    #student model
    encoder_output_patches_masked, grad_masking = self.mask_feature(encoder_output_patches)
    pos_enc = self.pos_enc(encoder_output_patches_masked).detach().to(device)
    encoder_output_patches_masked += pos_enc
    encoder_output_patches = self.decoder(encoder_output_patches)
    if init :
        print("masking feature attention finish", encoder_output_patches.shape)
        print("grad masking shape", grad_masking.shape)
    
    return encoder_output_patches, grad_masking
'''
        
def build_vitbackbone(args):
    tokenizer = None
    in_chan = args.in_chan
    #tokenizer = build_convnext(args, in_chans = in_chan)
    
    if args.model_scale == 't' :
        emb_dim = args.hidden_dim #256
        depth =  args.enc_layers
    if args.model_scale == 's' :
        patch_size = 2
        emb_dim = 512
        depth = 8
    if args.model_scale == 'm' :
        patch_size = 2
        emb_dim = 768
        depth = 12
    
    tokenizer = build_convnext(args, in_chans = in_chan, proj_dim =emb_dim)
    
    backbone = ViT(argument = args, batch = args.batch_size, in_chans = in_chan, kernel=args.kernel, emb_dim = emb_dim, depth = depth, block_stride = args.stride, tokenizer = tokenizer, group_idx = args.group_idx, pretrain = args.pretrain, rx_num_masking = args.rx_num_masking, tx_num_masking = args.tx_num_masking, mask_ratio = args.mask_ratio)
    
    return backbone

def build_selfsupervised_criterion(args):
    
    criterion = Self_Criterion(args)
    
    return criterion
    