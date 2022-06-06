#ifndef DISPATCH_H
#define DISPATCH_H

#include "session.h"
#include "graph.h"
#include "layer.h"

#ifdef __cplusplus
extern "C" {
#endif

void session_run(Session sess);

void forward_session(Session sess);
void backward_session(Session sess);

#ifdef __cplusplus
}
#endif

#endif