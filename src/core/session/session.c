#include "session.h"

void session_bind_graph(Session sess, Graph graph)
{
    sess.graph = graph;
}

void session_unbind_graph(Session sess)
{
    Graph graph = {0};
    sess.graph = graph;
}

void session_bind_data(Session sess)
{

}

void session_unbind_data(Session sess)
{

}

void session_run(Session sess)
{

}

void session_del(Session sess)
{
    
}