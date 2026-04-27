#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 

# Last modified : Wed 22 May 2019 08:10:00 PM EDT
# By Sabarish Sivanath
# To support Python 3

import os
import sys
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR

class Meteor:

    def __init__(self):
        d = dict(os.environ.copy())
        d['LANG'] = 'C'
        self.meteor_cmd = ['java', '-XX:+PerfDisableSharedMem', '-Xmx2G', '-jar', METEOR_JAR, \
                '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                cwd=os.path.dirname(os.path.abspath(__file__)), \
                stdin=subprocess.PIPE, \
                stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE, env=d)
        # Used to guarantee thread safety
        self.lock = threading.Lock()
        self._closed = False

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert (len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        # self.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
        self.meteor_p.stdin.write('{}\n'.format(eval_line).replace('.', ',').encode())
        self.meteor_p.stdin.flush()
        for i in range(0, len(imgIds)):
            scores.append(self._read_float())
        score = self._read_float()
        self.lock.release()

        return score, scores

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        score_line = score_line.replace('\n', '').replace('\r', '')
        self.meteor_p.stdin.write('{}\n'.format(score_line).replace('.', ',').encode())
        self.meteor_p.stdin.flush()
        raw = self._read_numeric_line()
        numbers = [str(int(float(n))) for n in raw.split()]
        return ' '.join(numbers)

    def method(self):
        return "METEOR"

    # def _stat(self, hypothesis_str, reference_list):
    #     # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    #     hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
    #     score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    #     self.meteor_p.stdin.write('{}\n'.format(score_line))
    #     return self.meteor_p.stdout.readline().strip()

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        stats = self._read_numeric_line()
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats 
        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        score = self._read_float()
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = self._read_float()
        self.lock.release()
        return score

    def _read_stdout_line(self):
        while True:
            raw = self.meteor_p.stdout.readline()
            if not raw:
                stderr_output = self.meteor_p.stderr.read().decode(errors='replace').strip()
                raise RuntimeError(f'METEOR process exited unexpectedly. stderr: {stderr_output}')
            line = raw.decode(errors='replace').strip()
            if not line:
                continue
            if self._is_jvm_warning(line):
                continue
            return line

    def _read_float(self):
        line = self._read_stdout_line()
        try:
            return float(line)
        except ValueError as exc:
            raise ValueError(f'Failed to parse METEOR float from line: {line!r}') from exc

    def _read_numeric_line(self):
        while True:
            line = self._read_stdout_line()
            try:
                for token in line.split():
                    float(token)
                return line
            except ValueError:
                continue

    @staticmethod
    def _is_jvm_warning(line):
        return line.startswith('[') and 'warning' in line.lower()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.lock.acquire()
        try:
            if self.meteor_p.stdin:
                self.meteor_p.stdin.close()
            self.meteor_p.kill()
            self.meteor_p.wait()
        finally:
            self.lock.release()
 
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
