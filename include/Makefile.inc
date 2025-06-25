# compiler flags, tools and include variables
GCC=g++
GCC_FLAGS=-Wall -Werror -pedantic -g
INCLUDES=-Iinclude
LINKS=

FORMATTER=clang-format
FORMATTER_FLAGS=-i

FORMATTER=clang-format
DOC=doxygen
PANDOC=pandoc
RM=rm -rf

FORMATTER_FLAGS=-i
DOC_FLAGS=config/Doxyfile
PANDOC_FLAGS=-s --variable geometry:margin=0.75in --variable urlcolor:NavyBlue --variable colorlinks:true
PANDOC_HIGHLIGHT=--highlight-style tango

SRC_DIR := src
OBJ_DIR := obj
DOC_DIR := docs
BIN_DIR := .
INC_DIR := include
TEST_TARGET=$(BIN_DIR)/test
SIM_TARGET=$(BIN_DIR)/sim


# sources and objects needed to be linked together for unit test executable
test_src := $(patsubst %.cpp, $(SRC_DIR)/%.cpp, $(test_src))
test_obj := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(test_src))
catch_test_obj := $(OBJ_DIR)/catch2-main.o
exception_obj := 

# objects needed to be linked together for main simulation executable
sim_src := $(patsubst %.cpp, $(SRC_DIR)/%.cpp, $(sim_src))
sim_obj := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(sim_src))

# pdf files for assignment description documentation
assg_doc := $(patsubst %.pdf, $(DOC_DIR)/%.pdf, $(assg_doc))

## List of all valid targets in this project:
## ------------------------------------------
## all          : by default generate all executables
##                (test and debug)
##
.PHONY : all
all : assgdocs


## assgdocs     : Create assignment description documentation from
##                README.md project markdown file.
##
.PHONY : assgdocs
assgdocs : $(assg_doc)

$(assg_doc) : README.md
	${PANDOC} ${PANDOC_FLAGS} ${PANDOC_HIGHLIGHT} -o $@ $^

## unit-tests   : Run all of the unit/doc tests available
##
.PHONY : unit-tests
unit-tests :
	python -m unittest test_assg_tasks.test_gru_music_generator_model
	python -m unittest test_assg_tasks.test_generate_jazz_solo
	python -m unittest test_assg_tasks.test_transformer_music_generator_model




	

## clean        : Remove auto-generated files for a completely
##                clean rebuild
##
.PHONY : clean
clean  :
	$(RM) *.o


## help         : Get all build targets supported by this build.
##
.PHONY : help
help : Makefile include/Makefile.inc
	@sed -n 's/^##//p' $^
