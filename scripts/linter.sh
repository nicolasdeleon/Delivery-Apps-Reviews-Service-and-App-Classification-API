#!/bin/bash

declare target="src"

echo "Running pylint & pycode style on $target"

pylint $target/*
pycodestyle $target/*

echo "Finished"
