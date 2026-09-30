import math
import torch
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp

def reset_parameters(named_parameters):
    for i in named_parameters():
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        else:
            nn.init.xavier_normal_(i[1])
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
from torch.nn import  Dropout



class Graph_Linear(nn.Module):##the linear layer
    def __init__(self,num_nodes, input_size, hidden_size, bias=True):
        super(Graph_Linear, self).__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.zeros(num_nodes,input_size,hidden_size))
        self.b = nn.Parameter(torch.zeros(num_nodes,hidden_size))
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x):
        output = torch.bmm(x.unsqueeze(1), self.W)
        output = output.squeeze(1)
        if self.bias:
            output = output + self.b
        return output

class Fuse_inlinear(nn.Module):
    def __init__(self,num_nodes,input_size=1):
        super(Fuse_inlinear,self).__init__()
        self.ww=nn.Parameter(torch.rand(num_nodes,input_size))
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self,x,y):
        output1=torch.mul(x,self.ww)+torch.mul(y,1-self.ww)
        return output1



class Graph_Tensor(nn.Module):##feature fusion for historical price and financial news
    def __init__(self, num_stock, d_hidden, d_market, d_news, bias=True):
        super(Graph_Tensor, self).__init__()
        self.num_stock = num_stock
        self.d_hidden = d_hidden
        self.d_market = d_market
        self.d_news = d_news
        self.seq_transformation_news = nn.Conv1d(d_news, d_hidden, kernel_size=1, stride=1, bias=False)
        self.seq_transformation_markets = nn.Conv1d(d_market, d_hidden, kernel_size=1, stride=1, bias=False)
        self.tensorGraph = nn.Parameter(torch.zeros(num_stock, d_hidden, d_hidden, d_hidden))
        self.W = nn.Parameter(torch.zeros(num_stock, 2 * d_hidden, d_hidden))
        self.b = nn.Parameter(torch.zeros(num_stock, d_hidden))
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, market, news):
        """
        [TODO] Fuse historical-market and financial-news signals with stock-specific tensor interactions.

        Input: market (time, stocks, d_market), news (time, stocks, d_news).
        Output: (time, stocks, d_hidden).
"""
        pass

class Graph_GRUCell(nn.Module):
    def __init__(self, num_nodes, input_size, hidden_size, bias=True):
        super(Graph_GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = Graph_Linear(num_nodes, input_size, 3 * hidden_size, bias=bias)
        self.h2h = Graph_Linear(num_nodes, hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x, hidden):
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy

class Graph_GRUModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, bias=True):
        super(Graph_GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = Graph_GRUCell(num_nodes, input_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(x.size()[1], self.hidden_dim, device=x.device,dtype = x.dtype)
        for seq in range(x.size(0)):
            hidden = self.gru_cell(x[seq], hidden)
        return hidden



class Graph_Attention(nn.Module):##Learning implicit relation

    def __init__(self, in_features, out_features, dropout, alpha ,alpha1,concat=True, residual=False):
        super(Graph_Attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.alpha1 = alpha1
        self.concat = concat
        self.residual = residual

        self.seq_transformation_r = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.seq_transformation_s = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)

        if self.residual:
            self.proj_residual = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)

        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)

        self.coef_revise = False
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def get_relation(self, input_r):
        num_stock = input_r.shape[0]
        seq_r = torch.transpose(input_r, 0, 1).unsqueeze(0)
        logits = torch.zeros(num_stock, num_stock, device=input_r.device, dtype=input_r.dtype)
        seq_fts_r = self.seq_transformation_r(seq_r)
        f_1 = self.f_1(seq_fts_r)
        f_2 = self.f_2(seq_fts_r)
        logits += (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
        coefs = F.elu(logits)
        coefs=F.softmax(coefs,dim=1)
        abc=torch.zeros_like(coefs)
        coefs = torch.where(coefs < self.alpha1, abc, coefs)
        if not isinstance(self.coef_revise,torch.Tensor):
            self.coef_revise = torch.zeros(73, 73, device = input_r.device) + 1.0 - torch.eye(73, 73,device = input_r.device)#note that:if you want to run this code on CSI300E,you should change 73 to 185(the number of firm nodes)
        coefs_eye = coefs.mul(self.coef_revise)
        return coefs_eye


    def forward(self, input_r,c):
        # unmasked attention
        coefs_eye = self.get_relation(input_r)*c
        coef=coefs_eye.nonzero(as_tuple=False)
        return coef


##dual attention networks

###intra-class attention
class RSMPConv(MessagePassing):
    def __init__(self, in_hid, out_hid, 
                 num_edge_types,negative_slope=0.2,heads=1):
        super(RSMPConv, self).__init__(aggr='add')

        self.in_hid = in_hid
        self.out_hid = out_hid
        self.num_edge_types = num_edge_types
        self.negative_slope=negative_slope
        
        self.rel_wi=nn.Parameter(torch.Tensor(num_edge_types,out_hid*2,1))
        self.rel_bt=nn.Parameter(torch.Tensor(out_hid*2,1))
        # self.w_wi=nn.Linear(in_hid, out_hid, bias=False)
        self.w_bt=nn.Linear(out_hid,out_hid,bias=False)
        self.q_trans=nn.Parameter(torch.Tensor(out_hid,1))

        self.norm=nn.LayerNorm(out_hid)
        self.norm_list=nn.ModuleList()
        for i in range(num_edge_types):
            self.norm_list.append(nn.LayerNorm(out_hid))


        self.skip = nn.Parameter(torch.ones(1))
        self.beta_weight=nn.Parameter(torch.ones(1))
        self.overall_beta=nn.Parameter(torch.randn(num_edge_types))
        # self.drop=Dropout(0.2)

        glorot(self.rel_wi)
        glorot(self.rel_bt)
        glorot(self.q_trans)

    def forward(self, x, edge_idx, edge_type):
        # x=self.w_wi(x)

        out_list=[]
        edg_list=[]
        for i in range(self.num_edge_types):
            mask = (edge_type == i)
            edge_index = edge_idx[:, mask]
            if mask.sum() !=0:
                rs=self.w_bt(F.leaky_relu(self.norm_list[i](self.propagate(edge_index, x=x,edge_type=i)),self.negative_slope))   #Nxd       
                out_list+=[rs]
                edg_list+=[i]
        beta=[]
        for i in edg_list:
            beta+=[F.leaky_relu(out_list[i]@self.q_trans,self.negative_slope).sum(0)]
        overall_beta=F.softmax(torch.FloatTensor(beta),dim=0)
        res=0
        for i in edg_list:
            res+=out_list[i]*overall_beta[i]
        
        final_weight=torch.sigmoid(self.skip)
        res = self.norm(F.gelu(res) * (final_weight) + x* (1 - final_weight))

        return res


    def message(self,edge_index,x_i, x_j,edge_type):
        
        node_f = torch.cat((x_i, x_j), 1)                                       #nx2d

        temp = torch.matmul(node_f, self.rel_wi[edge_type]).to(x_i.device)      #nx1

        alpha=softmax(temp,edge_index[1])
        rs=x_j*alpha                                                            #nxd
        return rs

###inter-class attention
class HetGATConv(MessagePassing):
    def __init__(self, in_hid, out_hid, negative_slope=0.2,norm=True,dual=True,global_weight=True):
        super(HetGATConv, self).__init__(aggr='add',)

        self.in_hid = in_hid
        self.out_hid = out_hid
        self.negative_slope=negative_slope
        self.norm=norm
        self.dual=dual
        self.global_weight=global_weight

        
        self.rel_wi=nn.Parameter(torch.Tensor(2,out_hid*2,1))
        self.rel_bt=nn.Parameter(torch.Tensor(out_hid*2,1))
        self.w_bt=nn.Linear(out_hid,out_hid,bias=False)
        self.w_out=nn.Linear(out_hid,out_hid,bias=False)
        self.q_trans=nn.Parameter(torch.Tensor(out_hid,1))

        self.out_norm=nn.LayerNorm(out_hid)

        self.skip = nn.Parameter(torch.ones(1))
        # self.drop=Dropout(0.2)

        glorot(self.rel_wi)
        glorot(self.rel_bt)
        glorot(self.q_trans)
        

    def forward(self, c_hid,p_hid, edge_idx, edge_type):
        out_list=[]
        num_edge_types=2
        edg_list=[]
        for i in range(num_edge_types):
            mask = (edge_type == i)
            edge_index = edge_idx[:, mask]
            if mask.sum() !=0:
                rs=self.w_bt(F.leaky_relu(self.propagate(edge_index=edge_index, x=(c_hid,p_hid),edge_type=i),self.negative_slope))   #Nxd
                out_list+=[rs]
                edg_list+=[i]
        beta=[]
        for i in edg_list:
            beta+=[F.leaky_relu(out_list[i]@self.q_trans,self.negative_slope).sum(0)]
        overall_beta=F.softmax(torch.FloatTensor(beta),dim=0)
        res=0
        for i in range(len(edg_list)):
            res+=out_list[i]*overall_beta[i]
        final_weight=torch.sigmoid(self.skip)
        res = self.out_norm(F.gelu(res)* (final_weight) + p_hid* (1 - final_weight))
        res=F.gelu(res)* (final_weight) + p_hid* (1 - final_weight)

        return res


    def message(self,x_i, x_j,edge_index,edge_type):

        node_f = torch.cat((x_i, x_j), 1)                                       #nx2d

        temp = torch.matmul(node_f, self.rel_wi[edge_type]).to(x_i.device)      #nx1

        alpha=softmax(temp,edge_index[1])
        #
        rs=x_j*alpha                                                            #nxd
        return rs

class SMPLayer(nn.Module):
    def __init__(self,in_hid,out_hid,num_m1,num_m2,n_heads=8,n_layers=2,dropout=0.2,hgt_layer=1,**kwargs):
        super(SMPLayer,self).__init__()
        self.hetgat=nn.ModuleList()
        self.layer=n_layers

        self.hgt=nn.ModuleList()
        self.norm=nn.LayerNorm(out_hid)
        self.drop=Dropout(dropout)
        self.proj_c=nn.Linear(in_hid,out_hid,bias=False)
        self.proj_p=nn.Linear(in_hid,out_hid,bias=False)

        for _ in range(hgt_layer):
            if _ == 0:
                self.hgt.append(RSMPConv(out_hid, out_hid, num_m1,heads=n_heads))
                self.hgt.append(RSMPConv(out_hid, out_hid, num_m2,heads=n_heads))
            else:
                self.hgt.append(RSMPConv(out_hid, out_hid, num_m1,heads=n_heads))
                self.hgt.append(RSMPConv(out_hid, out_hid, num_m2,heads=n_heads))

        for n in range(n_layers):
            self.hetgat.append(HetGATConv(out_hid, out_hid))
            self.hetgat.append(HetGATConv(out_hid, out_hid))
    #c_gh,p_gh:[x,edge_index,edge_type];t_gh[edge_index,edge_type]
    def forward(self,c_gh,p_gh,t_gh):
        h_c=c_gh[0]
        h_p=p_gh[0]
        h_c=self.proj_c(h_c)
        h_p=self.proj_p(h_p)
        edge_indx, edge_type = t_gh[0], t_gh[1]
        for ly in range(int(len(self.hetgat) / 2)):
            p_hid = self.hetgat[2 * ly](h_c, h_p, edge_indx, edge_type)

            edge_indx = torch.stack((edge_indx[1], edge_indx[0]))
            c_hid = self.hetgat[2 * ly + 1](h_p, h_c, edge_indx, edge_type)

            edge_indx = torch.stack((edge_indx[1], edge_indx[0]))
            h_c = c_hid
            h_p = p_hid
            
        for hl in range(int((len(self.hgt)/2))):
            # if hl==0:
            h_c=self.hgt[2*hl](h_c,c_gh[1],c_gh[2])
            h_p=self.hgt[2*hl+1](h_p,p_gh[1],p_gh[2])

        h_c=self.drop(self.norm(h_c))
        h_p=self.drop(self.norm(h_p))
        return h_c,h_p

class GraphCNN(nn.Module):
    def __init__(self,num_stock, d_market,d_news,out_c,d_hidden , hidn_rnn , hid_c, dropout ,alpha=0.2,alpha1=0.0054,t_mix=1,n_layeres=2,n_heads=1):##alpha1 denotes the normalized threshold
        super(GraphCNN, self).__init__()
        self.t_mix=t_mix
        self.dropout=dropout
        self.num_stock=num_stock
        if  self.t_mix == 0: # concat
            self.GRUs_s = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
            self.GRUs_r = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
        elif self.t_mix == 1: # all_tensor
             self.tensor = Graph_Tensor(num_stock,d_hidden,d_market,d_news)
             self.GRUs_s = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
             self.GRUs_r = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
        self.gcs=nn.ModuleList()
        self.project=nn.Sequential(
            nn.Linear(hidn_rnn,hid_c),
            nn.Tanh(),
            nn.Linear(hid_c,hid_c,bias=False))
        self.gcs=SMPLayer(hidn_rnn, hid_c,5,3,n_layers=1,dropout=0.2,hgt_layer=1)
        self.attentions = Graph_Attention(hidn_rnn, hid_c, dropout=dropout, alpha=alpha,alpha1=alpha1, residual=True, concat=True)
        self.X2Os = Graph_Linear(num_stock, hidn_rnn+hid_c , out_c, bias = True)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def get_relation(self,x_numerical, x_textual):
        x_r = self.tensor(x_numerical, x_textual)
        x_r = self.GRUs_r(x_r)
        relation = torch.stack([att.get_relation(x_r) for att in self.attentions])
        return relation

    def forward(self, x_market,x_news,edge_list,inter_metric,device1):
        if self.t_mix == 0:
            x_s = torch.cat([x_market, x_news], dim=-1)
            x_r = torch.cat([x_market, x_news], dim=-1)
        elif self.t_mix == 1:
            x_s = self.tensor(x_market, x_news)
            x_r = self.tensor(x_market, x_news)
        x_s = self.GRUs_s(x_s)
        x_r = self.GRUs_r(x_r)
        x = (inter_metric @ x_s)  # Generate the initial features of executive nodes
        edge_index=[]
        edge_dict={}
        edge_type=[]
        edge_index1=[]
        #the explicit firm relations
        for eg in ['I','B','S','SC']:
            if eg not in edge_dict:
                edge_dict[eg] = len(edge_dict)
            edge_index += edge_list[eg]
            edge_type += [torch.ones(len(edge_list[eg])) * edge_dict[eg]]
            edge_index1 += edge_list[eg]
        ##considering the implicit relation
        edge_index1=torch.LongTensor(edge_index1).transpose(0,1)
        edge_index1=edge_index1.cuda(device=device1)
        row,col=edge_index1.cpu()
        cc=torch.ones(row.shape[0]).cpu()
        d=1-(sp.coo_matrix((cc, (row, col)), shape=(self.num_stock, self.num_stock))).toarray()
        d[d<1]=0
        c = torch.from_numpy(d).cuda()
        c_adj= self.attentions(x_s, c)
        edge_index+=c_adj.tolist()
        company_list=c_adj.tolist()
        edge_type+=[torch.ones(len(company_list)) * 4]
        edge_type = torch.cat(edge_type, dim=0).cuda(device=device1)


        c_graph=[x_s,torch.LongTensor(edge_index).transpose(0,1).cuda(device=device1),edge_type]
        edge_index=[]
        edge_dict={}
        edge_type=[]
        ####the relations between firm and executives
        for eg in ['FS','FE']:
            if eg not in edge_dict:
                edge_dict[eg] = len(edge_dict)
            edge_index += edge_list[eg]
            edge_type += [torch.ones(len(edge_list[eg])) * edge_dict[eg]]
        edge_type=torch.cat(edge_type,dim=0).cuda(device=device1)
        t_graph=[torch.LongTensor(edge_index).transpose(0,1).cuda(device=device1),edge_type]
        edge_index=[]
        edge_dict={}
        edge_type=[]
        ##the explicit executives relatons
        for eg in ['R','C','CL']:
            if eg not in edge_dict:
                edge_dict[eg] = len(edge_dict)
            edge_index += edge_list[eg]
            edge_type += [torch.ones(len(edge_list[eg])) * edge_dict[eg]]
        edge_type=torch.cat(edge_type,dim=0).cuda(device=device1)
        p_graph=[x,torch.LongTensor(edge_index).transpose(0,1).cuda(device=device1),edge_type]
        x = F.dropout(x, self.dropout, training=self.training)
        x_c,x_p=self.gcs(c_graph,p_graph,t_graph)
        x_0 = torch.cat([x_s, x_c], dim=1)
        x_0 = F.elu(self.X2Os(x_0))
        out = F.log_softmax(x_0, dim=1)

        return out


if __name__ == "__main__":
    torch.manual_seed(42)
    passed = 0
    failed = 0
    def check(test_name, condition, detail=""):
        global passed, failed
        if bool(condition):
            passed += 1
            print(f"  [{test_name}] PASS")
        else:
            failed += 1
            suffix = f" - {detail}" if detail else ""
            print(f"  [{test_name}] FAIL{suffix}")
    def skip_checks(count, reason):
        global failed
        failed += count
        print(f"  [skipped {count} check(s)] FAIL - {reason}")
    print("DANSMP: 3-group automated test suite")
    try:
        module = Graph_Tensor(3, 4, 2, 3)
        output = module(torch.randn(2, 3, 2), torch.randn(2, 3, 3))
        check("Graph_Tensor output not None", output is not None)
        if output is not None:
            check("Graph_Tensor output shape", tuple(output.shape) == (2, 3, 4))
            check("Graph_Tensor output finite", torch.isfinite(output).all().item())
            output.sum().backward()
            check("Graph_Tensor multimodal gradients", module.tensorGraph.grad is not None and module.seq_transformation_news.weight.grad is not None)
        else: skip_checks(3, "Graph_Tensor returned None")
    except Exception as exc:
        print(f"  [Graph_Tensor] ERROR - {exc}"); skip_checks(4, "Graph_Tensor exception")
    try:
        module = Graph_GRUModel(4, 3, 5)
        output = module(torch.randn(3, 4, 3))
        check("Graph_GRUModel output not None", output is not None)
        if output is not None:
            check("Graph_GRUModel output shape", tuple(output.shape) == (4, 5))
            check("Graph_GRUModel output finite", torch.isfinite(output).all().item())
            output.sum().backward()
            check("Graph_GRUModel graph-specific gradients", module.gru_cell.x2h.W.grad is not None)
        else: skip_checks(3, "Graph_GRUModel returned None")
    except Exception as exc:
        print(f"  [Graph_GRUModel] ERROR - {exc}"); skip_checks(4, "Graph_GRUModel exception")
    try:
        module = Graph_Attention(5, 5, dropout=0.0, alpha=0.2, alpha1=0.0)
        output = module.get_relation(torch.randn(73, 5))
        check("Graph_Attention relation not None", output is not None)
        if output is not None:
            check("Graph_Attention relation shape", tuple(output.shape) == (73, 73))
            check("Graph_Attention relation finite", torch.isfinite(output).all().item())
            check("Graph_Attention removes self-relations", torch.allclose(torch.diagonal(output), torch.zeros(73)))
        else: skip_checks(3, "Graph_Attention returned None")
    except Exception as exc:
        print(f"  [Graph_Attention] ERROR - {exc}"); skip_checks(4, "Graph_Attention exception")
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    if failed != 0:
        raise SystemExit(1)