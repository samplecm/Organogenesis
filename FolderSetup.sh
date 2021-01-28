#!/bin/bash
#already have patients 1-4 in the file, I just want to rename the other ones starting at 5.
i=5
for file in $LOCATION/*
    do
        #check if name starts with SGF
        fileName="${file:0:3}"
        echo "${fileName}"
        if [$fileName -eq "SGB"]
            then
                mv "$file" "$P$i"
                i=i + 1
        fi
    done