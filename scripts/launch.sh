#!/bin/bash

export PYTHONPATH=.

exec $1 ${@:2}
