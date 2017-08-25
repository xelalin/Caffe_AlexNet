#!/bin/bash

pgrep run_   | xargs kill -9
pgrep python | xargs kill -9

