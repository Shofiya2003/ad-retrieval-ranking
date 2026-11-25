from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RetrieveRequest(_message.Message):
    __slots__ = ("query", "k", "exact")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    EXACT_FIELD_NUMBER: _ClassVar[int]
    query: str
    k: int
    exact: bool
    def __init__(self, query: _Optional[str] = ..., k: _Optional[int] = ..., exact: _Optional[bool] = ...) -> None: ...

class Hit(_message.Message):
    __slots__ = ("ad_id", "row", "score")
    AD_ID_FIELD_NUMBER: _ClassVar[int]
    ROW_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    ad_id: str
    row: int
    score: float
    def __init__(self, ad_id: _Optional[str] = ..., row: _Optional[int] = ..., score: _Optional[float] = ...) -> None: ...

class RetrieveResponse(_message.Message):
    __slots__ = ("hits", "bundle_version", "embed_us", "search_us")
    HITS_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_VERSION_FIELD_NUMBER: _ClassVar[int]
    EMBED_US_FIELD_NUMBER: _ClassVar[int]
    SEARCH_US_FIELD_NUMBER: _ClassVar[int]
    hits: _containers.RepeatedCompositeFieldContainer[Hit]
    bundle_version: str
    embed_us: int
    search_us: int
    def __init__(self, hits: _Optional[_Iterable[_Union[Hit, _Mapping]]] = ..., bundle_version: _Optional[str] = ..., embed_us: _Optional[int] = ..., search_us: _Optional[int] = ...) -> None: ...

class GetInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetInfoResponse(_message.Message):
    __slots__ = ("bundle_version", "model_name", "model_revision", "dim", "ad_count", "index_kind", "hnsw_m", "hnsw_ef_construction", "hnsw_ef_search_default")
    BUNDLE_VERSION_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_REVISION_FIELD_NUMBER: _ClassVar[int]
    DIM_FIELD_NUMBER: _ClassVar[int]
    AD_COUNT_FIELD_NUMBER: _ClassVar[int]
    INDEX_KIND_FIELD_NUMBER: _ClassVar[int]
    HNSW_M_FIELD_NUMBER: _ClassVar[int]
    HNSW_EF_CONSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    HNSW_EF_SEARCH_DEFAULT_FIELD_NUMBER: _ClassVar[int]
    bundle_version: str
    model_name: str
    model_revision: str
    dim: int
    ad_count: int
    index_kind: str
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search_default: int
    def __init__(self, bundle_version: _Optional[str] = ..., model_name: _Optional[str] = ..., model_revision: _Optional[str] = ..., dim: _Optional[int] = ..., ad_count: _Optional[int] = ..., index_kind: _Optional[str] = ..., hnsw_m: _Optional[int] = ..., hnsw_ef_construction: _Optional[int] = ..., hnsw_ef_search_default: _Optional[int] = ...) -> None: ...
