#!/usr/bin/env python
from math import exp, sqrt, log
import numpy
import pyEQ

class EnvelopeFollower(object):
  '''Calculates the instantaneous power envelope of a waveform using classical
  attack and release time parameters.'''
  
  def __init__(self, rate, windowSec, attackSec, releaseSec):
    '''Initializes and configures envelope detector.

    See configure() for parameters.'''
    self.configure(rate, windowSec, attackSec, releaseSec)

  def configure(self, rate, windowSec, attackSec, releaseSec):
    '''Configures detector for given parameters.

    rate - Input sample rate, in hertz.
    windowSec - The size of the evaluation window, in seconds.
    attackSec - Attack rate, in seconds.
    releaseSec - Release rate, in seconds.

    Attack and release rates are the measured times it takes for the
    difference between the follower's estimation of the input envelope and
    the actual input envelope to reach within negative one neper (about -4.3
    dB) of the input envelope.

    The attack rate is used whenever the current estimation is below the
    input and the release rate is used whenever the current estimation is
    above the input.'''
     
    # Compute the attack coefficient.
    self._ga = exp(-1/(rate*attackSec))
    # and the release coefficient.
    self._gr = exp(-1/(rate*releaseSec))
    self._windowSz = windowSec * rate
    self.reset()

  def reset(self):
    self._hist = numpy.zeros(self._windowSz, dtype='float')
    self._hist_i = 0
    self._envelope = 0.0

  def process(self, data):
    '''Calculate the instantaneous power envelope for the given array of
    input data, returning an output array of envelope values the same size.

    The values returned are power-scaled, not voltage scaled. Thus, if the
    input waveform has an RMS power of -10 dBfs, returns 0.01 and not
    0.0316. 0.316 would be the equivalent RMS voltage for the same signal.
    '''
    # Square each input sample and then rescale for the evaluation
    # window size. 
    data = data * data
    data = data / self._windowSz

    # Allocate the output array.
    ret = numpy.empty(len(data), dtype='float')

    # Cache some data that will be heavily used in the loop
    ga = self._ga
    gr = self._gr
    j = self._hist_i
    n = self._windowSz
    hist = self._hist
    sum = numpy.sum(self._hist)
    env = self._envelope

    for i, sample in enumerate(data):
        sum += sample
        sum -= hist[(j + i) % n]
        hist[(j + i) % n] = sample
        if env < sum:
          env *= ga
          env += (1 - ga) * sum
        else:
          env *= gr
          env += (1 - gr) * sum
        ret[i] = env

    self._envelope = env
    self._hist_i += len(data)
    self._hist_i %= n

    return ret

class Expander(object):
  '''Audio expander, using RMS sense. Once instantiated, can be configured
  for side-chain operation (signal A's amplitude is used to control signal B)
  or for direct operation (signal A is both the control and the input).'''

  def __init__(self, rate, window, attack, release, ratio, threshold, zero):
    '''Instantiate an RMS-sensitive audio expander.

    rate - Sample rate, in hertz.
    window - RMS averaging window size, in seconds.
    attack - Attack time, in seconds. (Neper scale).
    release - Release time, in seconds. (Neper scale).
    ratio - Expansion ratio, expressed as desired db(output level) /
            db(input level).
    threshold - Effect threshold, in dBfs (fs = "full scale", where full-scale
                is assumed to be a sample value of 1.0
    zero - Zero expansion point, in decibels. Input below this value will
           be made quieter and input above this value will be louder.'''
    self._detector = EnvelopeFollower(rate, window, attack, release)
    self._threshold_nepers = (threshold / 10.0) * log(10)
    #
    # We will be using a scalar multiplication of the input signal when we
    # apply the gain correction. To keep the expansion ratio as a power
    # ratio and not a voltage ratio, we must multiply by the square root
    # of the power gain required. This is easy to do in the logarithmic
    # domain by reducing expansion ratio by one half.
    # 
    self._ratio = (ratio - 1) / 2.0
    self._zero_nepers = ((zero - threshold)/ 10.0) * log(10)

  def process(self, data, control=None):
    '''Expand a buffer of samples, possibly using a different buffer of
    samples as the control (side-chain) channel.

    data - A buffer of samples to process.
    control - A buffer of samples to use as the side chain. If not
              specified, the data samples themselves will be used.'''
    if control is None:
      control = data
    envelope = self._detector.process(control)
    nepers = numpy.log(envelope)
    nepers -= self._threshold_nepers
    nepers[numpy.where(nepers < 0.0)] = 0.0
    nepers -= self._zero_nepers
    nepers *= self._ratio
    return data * numpy.exp(nepers)

class DBXDecoder(object):
  '''A decoder for the DBX noise reduction algorithm.'''

  Type_I = 1
  Type_II = 2 

  def __init__(self, typ, rate, zero_point_dbfs, threshold_dbfs, exp_r,
               neper_attack_time_s, neper_release_time_s):
    '''Instantiate a decoder for a given DBX algorithm and sample rate.

    typ - DBX algorithm/type, as an integer.
          2 = DBX type II (the only supported type at the moment).
    rate - Sampling rate, in hertz.
    zero_point_dbfs - Expander zero point, in dbFS.
    threshold_dbfs - Expander threshold, in dbFS.
    exp_r - Expansion ratio
    neper_attack_time_s - Attack time, in seconds, neper-basis.
    neper_attack_time_s - Release time, in seconds, neper-basis.'''

    if typ != self.Type_II:
      raise Exception('Only type II supported at the moment.')

    self._expander = Expander(
      rate,                 # sample rate
      0.001,                # RMS window time (seconds)
      neper_attack_time_s,  # Attack time (neper basis)
      neper_release_time_s, # Release time (neper basis)
      exp_r,                # Expansion ratio (out_dBfs/in_dBfs)
      threshold_dbfs,       # Expansion threshold (dBfs)
      zero_point_dbfs       # Expansion zero point (dBfs)
    )

    #
    # DBX's expansion is controlled by a filtered version of the
    # input signal. The filter is a specific equalizer and is known
    # as the "control EQ".
    #
    # The type I control EQ has three bands:
    #  According to Bob Weitz:
    #    1. A parametric cut filter with:
    #      * A 20 Hz center frequency
    #      * A Q value of .71
    #      * A gain of -8 dB
    #    2. A parametric boost filter with:
    #      * A center frequency of 20 kHz
    #      * A Q value of .78
    #      * A gain of 16 dB
    #    3. A low pass shelf filter with:
    #      * A cutoff frequency of 20 kHz
    #      * A cutoff rate of 12 dB/octave
    #  According to my reverse engineering of the MT100II deck:
    #    The DBX-II control curve:
    #    1. A parametric cut filter with:
    #      * A 20 Hz center frequency
    #      * A Q value of 2.25
    #      * A gain of -14.6 dB
    #    2. A parametric boost filter with:
    #      * A 7.8 kHz center frequency
    #      * A Q value of .58
    #      * A gain of 18 dB
    #    3. A low pass shelf filter with:
    #      * A 11.5 kHz cutoff frequency
    #      * A cutoff rate of 12 dB/octave
    #
    # In addition, the output of the control EQ chain is then reduced by
    # 3 dB. We will make this reduction happen in the process() method.
    #
    controlEQ = pyEQ.FilterChain()
    controlEQ._filters.append(
      pyEQ.Filter(
        pyEQ.FilterType.Peak,  # type
        20.0 / rate * 2,       # frequency (ratio against sample rate)
        -14.6,                 # gain (decibels)
        2.25,                  # q
      )
    )
    controlEQ._filters.append(
      pyEQ.Filter(
        pyEQ.FilterType.Peak,  # type
        7800.0 / rate * 2,     # frequency (ratio against sample rate)
        18.0,                  # gain (decibels)
        .58,                   # q
      )
    )
    controlEQ._filters.append(
      pyEQ.Filter(
        pyEQ.FilterType.LPButter, # type
        11500.0 / rate * 2,       # frequency (ratio against sample rate)
        0.0,                      # gain (decibels)
        1.,                       # (2^x * 6) db/ octave (if x=1 then db=12)
      )
    )
    self._controlEQ = controlEQ
    self._controlEQGain = 10 ** (-3/20.0)

    #
    # After expansion, the signal is treated with a post-emphasis filter.
    #
    # The post-emphasis filter has two bands:
    #  1. A parametric boost filter with:
    #    * A center frequency of 80 hertz
    #    * A Q value of .22
    #    * A gain value of 12 dB
    #
    postEQ = pyEQ.FilterChain()
    postEQ._filters.append(
      pyEQ.Filter(
        pyEQ.FilterType.Peak,  # type
        80.0 / rate * 2,       # frequency (ratio against sample rate)
        12.0,                  # gain (decibels)
        .22                    # q
      )
    )
    self._postEQ = postEQ

  def process(self, data):
    '''Decode the given data, returning the decoded data.'''
    control = self._controlEQ.filter(data)
    control *= self._controlEQGain
    expanded = self._expander.process(data, control)
    return self._postEQ.filter(expanded)
        
def main(args):
  import getopt
  import wave

  class WaveSource(object):
    def __init__(self, path):
      self._w = wave.open(path, 'r')
      self._rate = self._w.getframerate()
      self._channels = self._w.getnchannels()
      self._depth = self._w.getsampwidth()

    def rate(self):
      return self._rate

    def channels(self):
      return self._channels

    def depth(self):
      return self._depth

    def read(self, frame_count):
      '''Read next available samples from file.

      Returned data will be a numpy float multi-dimensional array.
      The first dimension of the array is the channel number. The second
      dimension is the sample number in the stream.

      All sample data is normalized to a dBfs value of 1.0, regardless of
      the input sample size.

      May return less frames than requested. If no input is available, returns
      None.'''
      data = self._w.readframes(frame_count)
      if data == '':
        return None

      channels = self._channels
      depth = self._depth
      frame_count = len(data) / channels / depth

      if depth == 1:
        samples = numpy.fromstring(data, dtype=numpy.int8)
        samples = samples.reshape((channels, frame_count), order='F')
        samples = samples / float(0x80)
      elif depth == 2:
        samples = numpy.fromstring(data, dtype=numpy.uint8)
        samples = samples.astype(numpy.int16)
        samples = (samples[1::2] << 8) + \
                  (samples[0::2]     )
        samples = samples.reshape((channels, frame_count), order='F')
        samples = samples / float(0x8000)
      else:
        assert(depth == 3)
        samples = numpy.fromstring(data, dtype=numpy.uint8)
        samples = samples.astype(numpy.int32)
        samples = (samples[2::3] << 24) + \
                  (samples[1::3] << 16) + \
                  (samples[0::3] << 8)
        samples = samples.reshape((channels, frame_count), order='F')
        samples = samples / float(0x80000000)

      return samples

  class WaveSink(object):
    def __init__(self, path, rate, channels, depth):
      '''Opens a WAVe file for writing.

      path - Pathname of file to write.
      rate - Sample rate, in hertz.
      channels - Number of channels.
      depth - Number of bytes per sample (1 = 8-bit, 2 = 16-bit, 3 = 24-bit )'''
      self._channels = channels
      self._depth = depth
      self._w = wave.open(path, 'w')
      self._w.setnchannels(channels)
      self._w.setframerate(rate)
      self._w.setsampwidth(depth)

    def write(self, frames):
      '''Writes samples to file.

      frames - An array of arrays containing sample data in float format.
               The outer array must contain one entry per channel. The inner
               array contains the sample data for that channel.  Values
               will be interpreted as 1.0 being 0 dBfs.'''
      framecount = len(frames[0])
      data = frames.reshape(self._channels*framecount, order='F')
      if self._depth == 1:
        data = (data * 0x80).clip(-0x80, 0x7f).astype(numpy.int8)
        data =  ''.join([chr(x) for x in data])
      elif self._depth == 2:
        data = (data * 0x8000).clip(-0x8000, 0x7fff).astype(numpy.int16)
        out = numpy.empty(2*framecount*self._channels, dtype=numpy.uint8)
        out[1::2] = (data >> 8) & 0xff
        out[0::2] = (data     ) & 0xff
        data = ''.join([chr(x) for x in out])
      else:
        assert(self._depth == 3)
        data = (data * 0x800000).clip(-0x800000, 0x7fffff).astype(numpy.int32)
        out = numpy.empty(3*framecount*self._channels, dtype=numpy.uint8)
        out[2::3] = (data >> 16) & 0xff
        out[1::3] = (data >>  8) & 0xff
        out[0::3] = (data      ) & 0xff
        data = ''.join([chr(x) for x in out])
      self._w.writeframes(data)

    def close(self):
      self._w.close()

  def usage(prog):
    print >>sys.stderr, 'usage:', prog, '[-y <dbx-type>] [-z <zero-point>] ' \
                        '[-R <ratio> [-a <attack>] [-r <release>] ' \
                        '[-t <threshold>]] <in-wav> <out-wav>'
    print >>sys.stderr, 'Process an audio file with DBX noise reduction.'
    print >>sys.stderr, '  -y <dbx-type>'
    print >>sys.stderr, '    Use the specified DBX decoding type. Valid types '
    print >>sys.stderr, '    are:'
    print >>sys.stderr, '      2 - DBX Type II (default)'
    print >>sys.stderr, '  -z <zero-point>'
    print >>sys.stderr, '    Set the expander\'s zero point level to <zero-point>'
    print >>sys.stderr, '    decibels, full-scale (dbFS). The default is -15.'
    print >>sys.stderr, '  -R <ratio>'
    print >>sys.stderr, '    The expansion ratio to use. (default = 2.0)'
    print >>sys.stderr, '  -a <attack_s>'
    print >>sys.stderr, '  -r <release_s>'
    print >>sys.stderr, '  -t <threshold_dbFS>'
    sys.exit(1)

  def error(msg, code):
    print >>sys.stderr, msg
    sys.exit(code)

  def fatal(msg):
    raise Exception(msg)

  prog = args[0]
  typ = DBXDecoder.Type_II

  #
  # According to Bob Weitz, DBX uses:
  #  - An expansion attack time of 16.5 ms (unknown log basis)
  #  - An expansion release time of 76 ms. (unknown log basis)
  #  - An expansion ratio of 2:1.
  #  - An expansion zero point of -15dB below tape saturation point
  #    We will assume that the input has been amplified such that the
  #    tape saturation point occurs at 0 dBfs.
  #
  # Over time I have changed these values by conducting my own tests of
  # each parameter on my own DBX unit and by establishing a definite basis
  # for the attack and release times in nepers. I now use these values:
  #
  exp_ratio = 2.05
  neper_attack_time = 0.049
  neper_release_time = 0.033
  zero_point_dbfs = -15
  threshold_dbfs = -61.0

  (opts, args) = getopt.getopt(args[1:], 'a:y:z:r:R:t:')
  for opt, val in opts:
    if opt == '-y':
      try:
        typ = int(val)
      except ValueError:
        error('Invalid DBX type', 1)
    elif opt == '-z':
      try:
        zero_point_dbfs = float(val)
      except ValueError:
        error('Invalid zero point', 1)
    elif opt == '-R':
      try:
        exp_ratio = float(val)
      except ValueError:
        error('Invalid expansion ratio', 1)
    elif opt == '-a':
      try:
        neper_attack_time = float(val)
      except ValueError:
        error('Invalid attack time', 1)
    elif opt == '-r':
      try:
        neper_release_time = float(val)
      except ValueError:
        error('Invalid release time', 1)
    elif opt == '-t':
      try:
        threshold_dbfs = float(val)
      except ValueError:
        error('Invalid threshold', 1)
    else:
      fatal('Oops. Unhandled option.')

  if typ != DBXDecoder.Type_II:
    error('Only DBX type II is supported at the moment.', 1)
        

  if len(args) < 2:
    usage(prog)

  in_file, out_file = args[0], args[1]  

  try:
    src = WaveSource(in_file)
  except:
    error('Can\'t open input file', 2)

  rate = src.rate()
  channels = src.channels()
   
  try:
    dst = WaveSink(out_file, rate, channels, src.depth())
  except:
    error('Can\'t open output file', 3)

  block_size = 2048
  out = numpy.empty((channels, block_size))

  processors = [
    DBXDecoder(
      typ,
      rate,
      zero_point_dbfs,
      threshold_dbfs,
      exp_ratio,
      neper_attack_time,
      neper_release_time,
    ) for x in range(channels)
  ]

  while True:
    frames = src.read(block_size)
    if frames is None:
      break

    if len(frames[0]) < block_size:
      out.resize((channels, len(frames[0])))

    for channel, samples in enumerate(frames):
      out[channel] = processors[channel].process(samples)
    dst.write(out)
    
if __name__ == '__main__':
  import sys
  main(sys.argv)
