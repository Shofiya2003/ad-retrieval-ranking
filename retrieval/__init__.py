"""Online retrieval: query text in, nearest ads out, served over gRPC.

Everything that touches a vector at request time lives here, next to the model
that embeds queries. The Go serving layer only ever sees ad ids and scores.
"""
