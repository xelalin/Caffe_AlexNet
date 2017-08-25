#!/bin/bash

ulimit -c 0 # disable core dump

echo "Starting demo..."
while true; do
  # kill any running processes
  kill -9 `pgrep -f "run_mp_.*"`
  sleep 1
  
  # remove ready signal file; start producer
  echo "Starting producer..."
  rm .producer_ready.txt
  ./run_mp_conv.sh &
  
  # start GUI
  ./run_demo_gui.sh > /dev/null 2>&1 &
  
  # wait for producer to be ready
  while ! test -f .producer_ready.txt; do
    sleep 1
  done
  
  # start consumer
  echo "Starting consumer..."
  ./run_mp_fc.sh &

  # monitor processes, break out to restart if necessary
  while pgrep -f run_mp_conv.sh; do
    sleep 1
  done

  echo "Restarting demo..."
done
