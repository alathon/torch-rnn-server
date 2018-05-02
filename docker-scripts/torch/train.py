# This is just a convenience script
# for easily running train.lua with reasonable
# arguments

import subprocess, argparse,os

parser = argparse.ArgumentParser()
parser.add_argument('--input_json', default='/data/generated/tiny-shakespeare.json')
parser.add_argument('--input_h5', default='/data/generated/tiny-shakespeare.h5')
parser.add_argument('--checkpoint_name', default='/data/checkpoints/checkpoint')
parser.add_argument('--debug_cmdline', type=bool, default=False)

# Specify overrides like so: --overrides -model_type rnn -rnn_size 256 ...
parser.add_argument('--overrides', nargs=argparse.REMAINDER)

args = parser.parse_args()

if __name__ == '__main__':
    overrides = ' '.join(args.overrides) if args.overrides else ''
    th = 'th /opt/torch/train.lua -input_json {} -input_h5 {}'.format(args.input_json, args.input_h5)
    if overrides:
        th = th + ' {}'.format(overrides)
    th = th.split(' ')
    if args.debug_cmdline: 
        print 'Running', th
    subprocess.call(th)
