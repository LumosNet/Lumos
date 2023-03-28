#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xor.h"

#define VERSION "0.1"

void lumos(int argc, char **argv)
{
    if (argc <= 1){
        fprintf(stderr, "Lumos fatal: No input option; use option --help for more information");
    }
    if (0 == strcmp(argv[1], "--version") || 0 == strcmp(argv[1], "-v"))
    {
        char version[] = VERSION;
        fprintf(stderr, "Lumos version: v%s\n", version);
        fprintf(stderr, "This is free software; see the source for copying conditions.  There is NO\n");
        fprintf(stderr, "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n");
    }
    else if (0 == strcmp(argv[1], "--demo") || 0 == strcmp(argv[1], "-d"))
    {
        if (argc <= 2){
            fprintf(stderr, "Lumos fatal: No such demo; use option --help for more information\n");
            return;
        }
        xor();
    }
    else if (0 == strcmp(argv[1], "--help") || 0 == strcmp(argv[1], "-h"))
    {
        fprintf(stderr, "Usage commands:\n");
        fprintf(stderr, "    --version or -v : To get version\n");
        fprintf(stderr, "    --path or -p : View The installation path\n");
        fprintf(stderr, "    --demo xor : To Run The xor demo\n");
        fprintf(stderr, "Thank you for using Lumos deeplearning framework.\n");
    }
    else
    {
        fprintf(stderr, "Lumos: \e[0;31merror\e[0m: unrecognized command line option '%s'\n", argv[1]);
        fprintf(stderr, "compilation terminated.\n");
        fprintf(stderr, "use --help or -h to get available commands.\n");
    }
}

int main(int argc, char **argv)
{
    lumos(argc, argv);
    return 0;
}
