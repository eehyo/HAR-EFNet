import torch
import torch.nn as nn

class EncoderBase(nn.Module):

    def __init__(self, args):

        super(EncoderBase, self).__init__()
        
        # 공통 파라미터
        self.input_channels = args['input_channels']     # 입력 채널 수 (보통 9: 가속도 x, y, z x 3부위)
        self.window_size = args['window_size']           # 입력 시계열 길이
        self.output_size = args['output_size']           # 출력 차원 (ECDF 특성 차원과 일치, 기본 234)
        self.device = args['device']
        
        # 인코더 타입 저장
        self.encoder_type = 'base'
    
    def forward(self, x):
        """
        forward pass (subclass implementation)
        Args:
            x: input data [batch_size, window_size, input_channels]
            
        Returns:
            encoder output [batch_size, output_size]
        """
        raise NotImplementedError("Subclasses must implement forward method") 
    
    def get_embedding_dim(self):

        return self.output_size 