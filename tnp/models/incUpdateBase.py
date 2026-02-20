# Interface slass to outline methods needed to be implemented for incremental context update support.
from abc import ABC, abstractmethod
import torch
import torch.distributions as td

# General inc updates
class IncUpdateEff(ABC):
    @abstractmethod
    def init_inc_structs(self, m: int, max_nc: int, device: str, use_flash: bool, cache_mhca: bool, persist_small: bool=False):
        raise NotImplementedError
    
    @abstractmethod
    def update_ctx(self, xc: torch.Tensor, yc: torch.Tensor, use_flash: bool, cache_mhca: bool, persist_small: bool=False):
        raise NotImplementedError
    
    @abstractmethod
    def repeat_ctx(self, repeat_times: int, persist_small: bool=False):
        raise NotImplementedError

    @abstractmethod
    def query(self, xt: torch.Tensor, dy: int, use_flash: bool, cache_mhca: bool, persist_small: bool=False) -> td.Normal:
        raise NotImplementedError


# This is the class for effecient incremental updates. It requires specification of knowledge of various known maximums.
# Incremental updating has been implemented for any size context, but this effecient version is just to make it even faster 
# for certain cases. Big O still the same and generic version tested. This is used for stuff like AR mode to make gains on small ctx
class IncUpdateEffFixed(ABC):
    @abstractmethod
    def init_inc_structs_fixed(self, m: int, max_nc: int, xt:torch.Tensor, device: str, use_flash: bool):
        raise NotImplementedError
    
    @abstractmethod
    def update_ctx_fixed(self, xc: torch.Tensor, yc: torch.Tensor, use_flash: bool):
        raise NotImplementedError

    @abstractmethod
    def query_fixed(self, tgt_start_ind: int, tgt_end_ind: int, use_flash: bool) -> td.Normal:
        raise NotImplementedError

