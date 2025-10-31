import torch
from torch import nn, Tensor
from typing import List



class VectorDynamicTanh(nn.Module):
    def __init__(self, input_shape):
    
        super().__init__()
        
           
        self.alpha = nn.Parameter(torch.randn(input_shape))
       

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x








class MLP(nn.Module):

    def __init__(self,dim):
        super().__init__()
        self.proj_1 =  nn.Linear(dim,dim,bias=False)
        self.proj_2 =  nn.Linear(dim,dim,bias=False)        
        self.gelu = nn.GELU()
       
             	   
    def forward(self, x):
       
        x = self.proj_1(x)
        x = self.gelu(x)          
        x = self.proj_2(x)
        
                          
        return x


     
class Task(nn.Module):
   

    def __init__(self, dim):
        super(Task, self).__init__()
       
        self.state_norm = VectorDynamicTanh(dim)
       
         
      

    def loss(self, f, x: Tensor) -> Tensor:
        state = self.state_norm(x)
        train_view =  state + torch.randn_like(state) 
        label_view = state
        return nn.functional.mse_loss(f(train_view), label_view)


class OGD(nn.Module):
  
    def __init__(self, lr: float = 0.01):
        super(OGD, self).__init__()
        self.lr = lr

    def step(self, model: nn.Module, grads: List[Tensor]):
      
        with torch.no_grad():
            for param, grad in zip(model.parameters(), grads):
              
                param -= self.lr * grad


class Learner(nn.Module):
   
    def __init__(self, task: Task, input_dim: int):
        super(Learner, self).__init__()
        self.task = task
        self.model = MLP(input_dim)
        self.probe_norm = VectorDynamicTanh(input_dim)
        self.optim = OGD()
       
    def train(self, x: Tensor):
       
        loss = self.task.loss(self.model, x)

        grad_fn = torch.autograd.grad(
            loss, self.model.parameters(),create_graph=True
        )

        self.optim.step(self.model, grad_fn)

    def predict(self, x: Tensor) -> Tensor:
       
        probe = self.probe_norm(x)
        return self.model(probe)


class TTT(nn.Module):
   

    def __init__(self, dim: int):
        super(TTT, self).__init__()
       
        self.task = Task(dim)

        self.learner = Learner(self.task,dim)

    def forward(self, in_seq: Tensor) -> Tensor:

       
        outs = []
        
        for seq in range(in_seq.size(1)):
           
            self.learner.train(in_seq[:,seq,:])
            pred = self.learner.predict(in_seq[:,seq,:]).detach()
            outs.append(pred)
        out = torch.stack(outs, dim=1)
        
        return out
        

class LocalMappingUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()
        
           
        self.token_norm = VectorDynamicTanh(dim)
        self.mlp = MLP(dim)
      
             	   
    def forward(self, x):
    
        x = self.token_norm(x)
        x = self.mlp(x)     	
      
        
        return x
    	

class GlobalMappingUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()
        
             
       
        self.ttt = TTT(dim)       
        
              
                                      	   
    def forward(self, x):
    
               
        x = self.ttt(x)
        
        return x         




class TTTBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
       
         
        self.local_mapping = LocalMappingUnit(d_model)
        self.global_mapping = GlobalMappingUnit(d_model)
        
    
        
        
        
    def forward(self, x):
                  
        residual = x
        
        x = self.global_mapping(x)
    
        x = x + residual
        
        residual = x
        
        x = self.local_mapping(x)
        
                                          
        out = x + residual
        
        
        return out



class TTTM(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        
        self.model = nn.Sequential(
            *[TTTBlock(d_model) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








