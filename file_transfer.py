#!/usr/bin/env python3

from optparse import OptionParser
import os
from subprocess import Popen
from multiprocessing import Process, Queue
from tqdm import tqdm
import contextlib
import sys

parser = OptionParser()
parser.add_option("-b", "--basedir", dest="basedir", type="string",
                  help="[REQUIRED] root folder to transfer")
parser.add_option("-u", "--user", dest="user", type="string",
                  help="[REQUIRED] User from host")
parser.add_option("-s", "--host", dest="host", type="string",
                  help="[REQUIRED] Host to scp to")
parser.add_option("-p", "--path", dest="path", type="string",
                  help="[REQUIRED] path on the host")
parser.add_option("-n", "--threads", dest="threads", type="int", default=1,
                  help="[OPTIONAL] number of parallel transfers")
parser.add_option("-t", "--tree", dest="tree", action="store_true", default=False,
                  help="[OPTIONAL] create the tree structure.")
parser.add_option("-c", "--copy", dest="copy", action="store_true", default=False,
                  help="[OPTIONAL] copy the files.")
parser.add_option("-d", "--debug", dest="debug", action="store_true", default=False,
                  help="[OPTIONAL] Run debug mode, will not perform any action.")

(options, args) = parser.parse_args()

class DummyFile(object):
  file = None
  def __init__(self, file):
    self.file = file

  def write(self, x):
    # Avoid print() second call (useless \n)
    if len(x.rstrip()) > 0:
        tqdm.write(x, file=self.file)

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout

def make_directory(args):
    user, host, path = args
    cmd = 'ssh {}@{} bash -c "mkdir -p {}"'.format(user, host, path)
    cmd = cmd.split(' ')
    p = Popen(cmd)
    return p

def transfer_file(args):
    user, host, path, filepath = args
    cmd = 'scp {} {}@{}:{}'.format(filepath, user, host, path)
    cmd = cmd.split(' ')
    p = Popen(cmd)
    return p

def worker(queue):
    while not queue.empty():
        action = queue.get(True, 2)
        command = action[0]
        args = action[1]
        if command == "mkdir":
            p = make_directory(args)
        elif command == 'copy':
            p = transfer_file(args)
        status = p.wait()

def explore(mkQ, cpQ, path):
    ls = os.listdir(path)
    print("Now entering: ", path)
    for el in ls:
        el = os.path.join(path, el)
        if os.path.isdir(el):
            hostpath = el.replace(options.basedir, options.path)
            mkQ.put(['mkdir', (options.user, options.host, hostpath)])
            explore(mkQ, cpQ, el)
        elif os.path.isfile(el):
            hostpath = el.replace(options.basedir, options.path)
            cpQ.put(['copy', (options.user, options.host, hostpath, el)])

if __name__ == "__main__":
    mkQ = Queue()
    cpQ = Queue()
    print("Reading the file tree and preparing the actions to be performed")
    explore(mkQ, cpQ, options.basedir)
    print("mkQ: {} \t cpQ: {}".format(mkQ.qsize(), cpQ.qsize()))
    if options.debug:
        print("Debugging ON !")
        while not mkQ.empty():
            action = mkQ.get_nowait()
            user, host, path = action[1]
            cmd = 'ssh {}@{} "mkdir -p {}"'.format(user, host, path)
            print("Making Directory: {}".format(cmd))
        while not cpQ.empty():
            action = cpQ.get_nowait()
            user, host, path, filepath = action[1]
            cmd = 'scp {} {}@{}:{}'.format(filepath, user, host, path)
            print("Copying file: {}".format(cmd))
    else:
        if options.tree:
            print("Starting to make directory tree...")
            workers = {}
            tot = mkQ.qsize()
            lasttot = tot
            for i in range(options.threads):
                workers[i] = Process(target=worker, args=(mkQ,))
                workers[i].start()
            with tqdm(total=tot, file=sys.stdout) as bar:
                while True:
                    newtot = mkQ.qsize()
                    if newtot != lasttot:
                        bar.update(lasttot-newtot)
                        lasttot = newtot
                    if mkQ.empty():
                        break
                tqdm.write("Waiting for all the workers to finish ...")
                for i in range(options.threads):
                    workers[i].join()
                    bar.update(1)
        if options.copy:
            print("Now starting the file transfer !")
            workers = {}
            tot = cpQ.qsize()
            lasttot = tot
            for i in range(options.threads):
                workers[i] = Process(target=worker, args=(cpQ,))
                workers[i].start()
            with tqdm(total=tot, file=sys.stdout) as bar:
                while True:
                    newtot = cpQ.qsize()
                    if newtot != lasttot:
                        bar.update(lasttot-newtot)
                        lasttot = newtot
                    if cpQ.empty():
                        break
                tqdm.write("Waiting for all the workers to finish ...")
                for i in range(options.threads):
                    workers[i].join()
                    bar.update(1)
