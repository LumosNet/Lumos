#include <stdio.h>
#include <stdlib.h>

#include "tsession.h"

void lumos_t(int argc, char **argv)
{
    if (argc <= 1){
        fprintf(stderr, "Lumos_t fatal: No input option; use option --help for more information")
    }
    if (0 == strcmp(argv[1], "--version") || 0 == strcmp(argv[1], "-v"))
    {

    } else if (0 == strcmp(argv[1], "--help" || 0 == strcmp(argv[1], "-h")){
    
    } else {
        fprintf(stderr, "Lumos_t: \e[0;31merror\e[0m: unrecognized command line option '%s'\n", argv[1]);
        fprintf(stderr, "compilation terminated.\n");
        fprintf(stderr, "use --help or -h to get available commands.\n");
    }
}

int main(int argc, char *argv[])
{
    lumos_t(argc, argv);
    return 0;
}
