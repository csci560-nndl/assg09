import numpy as np
import copy
import random
import math
# NOTE: Version 6.7.1 of music21 is being used here, V7+ seem to have changed API some and the below code
# no longer works for more recent versions of the music21 package.
from music21 import converter, instrument, key, stream, tempo, meter, converter, note, chord, interval, scale, midi
from mido import MidiFile
from pydub import AudioSegment
from pydub.generators import Sine
from collections import OrderedDict, defaultdict
from itertools import groupby, zip_longest


def load_music_dataset(file):
    """Load the music "values" dataset from given .mid midi file.

    Parameters
    ----------
    file : str
        Name of file of midi musical values data

    Returns
    -------
    X : ndarray shape (num_samples, sequence_length, vocab_size)
        The training input data, sequence length is 30 time steps of values, vocabulary size has 90 different musical values
    Y : ndarray shape (num_samples, sequence_length, vocab_size)
        The training labels, just shifted 1 time step, otherwise same as training input
    vocab_size : int
        The number of "values" in our music vocabulary, should be 90
    vocab_dict : dict
        The vocabulary dictionary, maps an index (key 0-89) to a musical value
    chords : 
        Chords used in the input midi
    """
    chords, abstract_grammars = get_musical_data(file)
    corpus, _, inverse_vocab_dict, vocab_dict = get_corpus_data(abstract_grammars)
    X, Y, vocab_size = data_processing(corpus, inverse_vocab_dict, 60, 30)   
    return (X, Y, vocab_size, vocab_dict, chords)


def get_musical_data(file ):
    """Get musical data from a midi file.  We first parse the midi file
    to get its measures and cords, then turn that into an abstract grammar

    Parameters
    ----------
    file : str
        Name of file of midi musical values data

    Returns
    -------
    chords : dict like 
        A dict like collection of musical chords from midi file
    abstract_grammars :
        The abstract grammar of the musical composition
    """
    measures, chords = parse_midi(file)
    abstract_grammars = get_abstract_grammars(measures, chords)
    return chords, abstract_grammars


def get_corpus_data(abstract_grammars):
    """Given an abstract grammar of a musical composition, decompose into
    the corpus data we need for training sequenc-to-sequence generators.

    Parameters
    ----------
    abstract_grammars :
        The abstract grammar of the musical composition
    
    Returns
    -------
    corpus : sequence
        A musical corpus, basicicall a sequence of musical values
    values : set
        The unique musical values in the corpus, basically the vocabulary of our
        corpus.
    inverse_vocab_dict: dict
        A dictionary that maps a musical value to its assigned integer index of the
        vocabulary
    vocab_dict: dict
        A dictionary that maps a the integer index of a vocabulary to its
        musical value
    """
    corpus = [x for sublist in abstract_grammars for x in sublist.split(' ')]
    values = set(corpus)
    inverse_vocab_dict = dict((value, index) for index, value in enumerate(values))
    vocab_dict = dict((index, value) for index, value in enumerate(values))
    return corpus, values, inverse_vocab_dict, vocab_dict


def get_abstract_grammars(measures, chords):
    """Extract grammars from musical measures and chords sequences

    Parameters
    ----------
    measures : sequence
        A sequence like series of musical measures
    chords : dict like 
        A dict like collection of musical chords from midi file

    Returns
    -------
    abstract_grammars :
        The abstract grammar of the musical composition
    """
    abstract_grammars = []
    for ix in range(1, len(measures)):
        m = stream.Voice()
        for i in measures[ix]:
            m.insert(i.offset, i)
        c = stream.Voice()
        for j in chords[ix]:
            c.insert(j.offset, j)
        parsed = parse_melody(m, c)
        abstract_grammars.append(parsed)

    return abstract_grammars


def data_processing(corpus, inverse_vocab_dict, num_samples = 60, sequence_length = 30):
    """Create numpy array suitable for deep network training from corpus of
    musical values and their assigned one-hot encoding index dictionary.

    Parameters
    ----------
    corpus : sequence
        A musical corpus, basicicall a sequence of musical values
    inverse_vocab_dict: dict
        A dictionary that maps a musical value to its assigned integer index of the
        vocabulary
    num_samples : int
        Number of samples in dataset, default 60
    sequence_length : int
        Length of timeseries of each sample, default 30

    Returns
    -------
    X, Y : ndarray shape (num_samples, sequence_length, vocab_size)
        Numpy arrays of one-hot encoded musical sequences
    vocab_size : int 
        Total number of musical values in the vocabulary 
    """
    vocab_size = len(set(corpus))
    np.random.seed(0)
    X = np.zeros((num_samples, sequence_length, vocab_size))
    Y = np.zeros((num_samples, sequence_length, vocab_size))

    # choose a sample of size sequence_length from a random start:end
    # of the whole corpus sequence for out training samples
    print('data processing len(corpus):', len(corpus))
    for sample_idx in range(num_samples):
        random_idx = np.random.choice(len(corpus) - sequence_length)
        corpus_data = corpus[random_idx:(random_idx + sequence_length)]
        for sequence_idx in range(sequence_length):
            value_idx = inverse_vocab_dict[corpus_data[sequence_idx]]
            if sequence_idx != 0:
                X[sample_idx, sequence_idx, value_idx] = 1
                Y[sample_idx, sequence_idx-1, value_idx] = 1
    
    return X, Y, vocab_size 


def parse_melody(full_measure_notes, full_measure_chords):
    """Parse melody from notes and chords

    Parameters
    ----------
    full_measure_notes : 
    full_measure_chords

    Returns
    -------
    """
    # Remove extraneous elements.x
    measure = copy.deepcopy(full_measure_notes)
    chords = copy.deepcopy(full_measure_chords)
    measure.removeByNotOfClass([note.Note, note.Rest])
    chords.removeByNotOfClass([chord.Chord])

    # Information for the start of the measure.
    # 1) measure_start_time: the offset for measure's start, e.g. 476.0.
    # 2) measure_start_offset: how long from the measure start to the first element.
    measure_start_time = measure[0].offset - (measure[0].offset % 4)
    #measure_start_offset  = measure[0].offset - measure_start_time

    # Iterate over the notes and rests in measure, finding the grammar for each
    # note in the measure and adding an abstract grammatical string for it. 

    full_grammar = ""
    prev_note = None # Store previous note. Need for interval.
    num_non_rests = 0 # Number of non-rest elements. Need for updating prev_note.
    for ix, nr in enumerate(measure):
        # Get the last chord. If no last chord, then (assuming chords is of length
        # >0) shift first chord in chords to the beginning of the measure.
        try: 
            last_chord = [n for n in chords if n.offset <= nr.offset][-1]
        except IndexError:
            chords[0].offset = measure_start_time
            last_chord = [n for n in chords if n.offset <= nr.offset][-1]

        # FIRST, get type of note, e.g. R for Rest, C for Chord, etc.
        # Dealing with solo notes here. If unexpected chord: still call 'C'.
        element_type = ' '
        # R: First, check if it's a rest. Clearly a rest --> only one possibility.
        if isinstance(nr, note.Rest):
            element_type = 'R'
        # C: Next, check to see if note pitch is in the last chord.
        elif nr.name in last_chord.pitchNames or isinstance(nr, chord.Chord):
            element_type = 'C'
        # L: (Complement tone) Skip this for now.
        # S: Check if it's a scale tone.
        elif is_scale_tone(last_chord, nr):
            element_type = 'S'
        # A: Check if it's an approach tone, i.e. +-1 halfstep chord tone.
        elif is_approach_tone(last_chord, nr):
            element_type = 'A'
        # X: Otherwise, it's an arbitrary tone. Generate random note.
        else:
            element_type = 'X'

        # SECOND, get the length for each element. e.g. 8th note = R8, but
        # to simplify things you'll use the direct num, e.g. R,0.125
        if (ix == (len(measure)-1)):
            # formula for a in "a - b": start of measure (e.g. 476) + 4
            diff = measure_start_time + 4.0 - nr.offset
        else:
            diff = measure[ix + 1].offset - nr.offset

        # Combine into the note info.
        note_info = "%s,%.3f" % (element_type, nr.quarterLength) # back to diff

        # THIRD, get the deltas (max range up, max range down) based on where
        # the previous note was, +- minor 3. Skip rests (don't affect deltas).
        interval_info = ""
        if isinstance(nr, note.Note):
            num_non_rests += 1
            if num_non_rests == 1:
                prev_note = nr
            else:
                note_dist = interval.Interval(noteStart=prev_note, noteEnd=nr)
                note_dist_upper = interval.add([note_dist, "m3"])
                note_dist_lower = interval.subtract([note_dist, "m3"])
                interval_info = ",<%s,%s>" % (note_dist_upper.directedName, 
                    note_dist_lower.directedName)
                prev_note = nr

        # Return. Do lazy evaluation for real-time performance.
        grammar_term = note_info + interval_info 
        full_grammar += (grammar_term + " ")

    return full_grammar.rstrip()


def parse_midi(file):
    """Parse the MIDI data for separate melody and accompaniment parts.
    Parameters
    ----------
    file : str
        Name of file of midi musical values data

    Returns
    -------
    measures : sequence
        A sequence like series of musical measures
    chords : dict like 
        A dict like collection of musical chords from midi file
    """
    midi_data = converter.parse(file)

    # Get melody part, compress into single voice.
    melody_stream = midi_data[5]     # For Metheny piece, Melody is Part #5.
    melody1, melody2 = melody_stream.getElementsByClass(stream.Voice)
    for j in melody2:
        melody1.insert(j.offset, j)
    melody_voice = melody1

    for i in melody_voice:
        if i.quarterLength == 0.0:
            i.quarterLength = 0.25

    # Change key signature to adhere to comp_stream (1 sharp, mode = major).
    # Also add Electric Guitar. 
    melody_voice.insert(0, instrument.ElectricGuitar())
    melody_voice.insert(0, key.KeySignature(sharps=1))

    # The accompaniment parts. Take only the best subset of parts from
    # the original data. Maybe add more parts, hand-add valid instruments.
    # Should add least add a string part (for sparse solos).
    # Verified are good parts: 0, 1, 6, 7 '''
    partIndices = [0, 1, 6, 7]
    comp_stream = stream.Voice()
    comp_stream.append([j.flat for i, j in enumerate(midi_data) 
        if i in partIndices])

    # Full stream containing both the melody and the accompaniment. 
    # All parts are flattened. 
    full_stream = stream.Voice()
    for i in range(len(comp_stream)):
        full_stream.append(comp_stream[i])
    full_stream.append(melody_voice)

    # Extract solo stream, assuming you know the positions ..ByOffset(i, j).
    # Note that for different instruments (with stream.flat), you NEED to use
    # stream.Part(), not stream.Voice().
    # Accompanied solo is in range [478, 548)
    solo_stream = stream.Voice()
    for part in full_stream:
        curr_part = stream.Part()
        curr_part.append(part.getElementsByClass(instrument.Instrument))
        curr_part.append(part.getElementsByClass(tempo.MetronomeMark))
        curr_part.append(part.getElementsByClass(key.KeySignature))
        curr_part.append(part.getElementsByClass(meter.TimeSignature))
        curr_part.append(part.getElementsByOffset(476, 548, 
                                                  includeEndBoundary=True))
        cp = curr_part.flat
        solo_stream.insert(cp)

    # Group by measure so you can classify. 
    # Note that measure 0 is for the time signature, metronome, etc. which have
    # an offset of 0.0.
    melody_stream = solo_stream[-1]
    measures = OrderedDict()
    offsetTuples = [(int(n.offset / 4), n) for n in melody_stream]
    measureNum = 0 # for now, don't use real m. nums (119, 120)
    for key_x, group in groupby(offsetTuples, lambda x: x[0]):
        measures[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Get the stream of chords.
    # offsetTuples_chords: group chords by measure number.
    chordStream = solo_stream[0]
    chordStream.removeByClass(note.Rest)
    chordStream.removeByClass(note.Note)
    offsetTuples_chords = [(int(n.offset / 4), n) for n in chordStream]

    # Generate the chord structure. Use just track 1 (piano) since it is
    # the only instrument that has chords. 
    # Group into 4s, just like before. 
    chords = OrderedDict()
    measureNum = 0
    for key_x, group in groupby(offsetTuples_chords, lambda x: x[0]):
        chords[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Fix for the below problem.
    #   1) Find out why len(measures) != len(chords).
    #   ANSWER: resolves at end but melody ends 1/16 before last measure so doesn't
    #           actually show up, while the accompaniment's beat 1 right after does.
    #           Actually on second thought: melody/comp start on Ab, and resolve to
    #           the same key (Ab) so could actually just cut out last measure to loop.
    #           Decided: just cut out the last measure. 
    del chords[len(chords) - 1]
    assert len(chords) == len(measures)

    return measures, chords


def is_scale_tone(chord, note):
    """Generate all scales that have the chord notes then check if note is
    in names.

    Parameters
    ----------
    chord :
    note : 
        The note to check if it is a scale tone

    Returns
    -------
    bool : true if the note is a scale tone, false if not
    """
    # Derive major or minor scales (minor if 'other') based on the quality
    # of the chord.
    scale_type = scale.DorianScale() # i.e. minor pentatonic
    if chord.quality == 'major':
        scale_type = scale.MajorScale()
    # Can change later to deriveAll() for flexibility. If so then use list
    # comprehension of form [x for a in b for x in a].
    scales = scale_type.derive(chord) # use deriveAll() later for flexibility
    all_pitches = list(set([pitch for pitch in scales.getPitches()]))
    all_note_names = [i.name for i in all_pitches] # octaves don't matter

    # Get note name. Return true if in the list of note names.
    noteName = note.name
    return (noteName in all_note_names)


def is_approach_tone(chord, note):
    """See if note is +/- 1 a chord tone.
    Parameters
    ----------
    chord :
    note : 
        The note to check if it is a scale tone

    Returns
    -------
    bool : true if the note is an approach tone, false if not
    """
    for chordPitch in chord.pitches:
        stepUp = chordPitch.transpose(1)
        stepDown = chordPitch.transpose(-1)
        if (note.name == stepDown.name or 
            note.name == stepDown.getEnharmonic().name or
            note.name == stepUp.name or
            note.name == stepUp.getEnharmonic().name):
                return True
    return False


def generate_music(file, solo, vocab_dict, chords, diversity = 0.5):
    """
    Generates music using a model trained to learn musical patterns of a jazz soloist. Creates an audio stream
    to save the music and play it.
    
    Arguments
    ---------
    file : str
        The name of the output file to generate file to.
    solo : ndarray shape (sequence_length, )
        A vector of musical value indexes from the musical vocabulary
    vocab_dict : ndarray
        A python dictionary mapping indices (0-89) into their corresponding unique tone (ex: A,0.250,< m2,P-4 >)
    chords :
    diversity :
    
    Returns
    -------
    out_stream : stream 
        A music21 output stream, with which we can?
    """
    # set up audio stream
    out_stream = stream.Stream()
    
    # Initialize chord variables
    curr_offset = 0.0                                     # variable used to write sounds to the Stream.
    num_chords = int(len(chords) / 3)                     # number of different set of chords
    
    print("Predicting new values for different set of chords.")
    # Loop over all 18 set of chords. At each iteration generate a sequence of tones
    # and use the current chords to convert it into actual sounds 
    for i in range(1, num_chords):
        
        # Retrieve current chord from stream
        curr_chords = stream.Voice()
        
        # Loop over the chords of the current set of chords
        for j in chords[i]:
            # Add chord to the current chords with the adequate offset, no need to understand this
            curr_chords.insert((j.offset % 4), j)
        
        # Generate a sequence of tones using the model
        #_, indices = predict_and_sample(inference_model)
        indices = list(solo.squeeze())
        pred = [vocab_dict[p] for p in indices]
        
        predicted_tones = 'C,0.25 '
        for k in range(len(pred) - 1):
            predicted_tones += pred[k] + ' ' 
        
        predicted_tones +=  pred[-1]
                
        #### POST PROCESSING OF THE PREDICTED TONES ####
        # We will consider "A" and "X" as "C" tones. It is a common choice.
        predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')

        # Pruning #1: smoothing measure
        predicted_tones = prune_grammar(predicted_tones)
        
        # Use predicted tones and current chords to generate sounds
        sounds = unparse_grammar(predicted_tones, curr_chords)

        # Pruning #2: removing repeated and too close together sounds
        sounds = prune_notes(sounds)

        # Quality assurance: clean up sounds
        sounds = clean_up_notes(sounds)

        # Print number of tones/notes in sounds
        print('Generated %s sounds using the predicted values for the set of chords ("%s") and after pruning' % (len([k for k in sounds if isinstance(k, note.Note)]), i))
        
        # Insert sounds into the output stream
        for m in sounds:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0
        
    # Initialize tempo of the output stream with 130 bit per minute
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    # Save audio stream to file
    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open(file, 'wb')
    mf.write()
    print(f"Your generated music is saved in {file}")
    mf.close()
    
    # Play the final stream through output (see 'play' lambda function above)
    # play = lambda x: midi.realtime.StreamPlayer(x).play()
    # play(out_stream)
    
    return out_stream

def mid2wav(file):
    """Given a .midi file, convert it to a .wav file, so can be played with basic
    iPython.display.Audio()

    Parameters
    ----------
    file : str
        The name of the input file to render into a wave.

    Returns
    -------
    None, but new file named rendered.wav will be created from the midi file
    """
    mid = MidiFile(file)
    output = AudioSegment.silent(mid.length * 1000.0)

    tempo = 130 # bpm

    for track in mid.tracks:
        # position of rendering in ms
        current_pos = 0.0
        current_notes = defaultdict(dict)

        for msg in track:
            current_pos += ticks_to_ms(msg.time, tempo, mid)
            if msg.type == 'note_on':
                if msg.note in current_notes[msg.channel]:
                    current_notes[msg.channel][msg.note].append((current_pos, msg))
                else:
                    current_notes[msg.channel][msg.note] = [(current_pos, msg)]


            if msg.type == 'note_off':
                start_pos, start_msg = current_notes[msg.channel][msg.note].pop()

                duration = math.ceil(current_pos - start_pos)
                signal_generator = Sine(note_to_freq(msg.note, 500))
                #print(duration)
                rendered = signal_generator.to_audio_segment(duration=duration-50, volume=-20).fade_out(100).fade_in(30)

                output = output.overlay(rendered, start_pos)

    output.export("../output/rendered.wav", format="wav")

def note_to_freq(note, concert_A=440.0):
  '''
  from wikipedia: http://en.wikipedia.org/wiki/MIDI_Tuning_Standard#Frequency_values
  '''
  return (2.0 ** ((note - 69) / 12.0)) * concert_A

def ticks_to_ms(ticks, tempo, mid):
    tick_ms = math.ceil((60000.0 / tempo) / mid.ticks_per_beat)
    return ticks * tick_ms


def unparse_grammar(m1_grammar, m1_chords):
    """Given a grammar string and chords for a measure, returns measure notes.
    """
    m1_elements = stream.Voice()
    currOffset = 0.0 # for recalculate last chord.
    prevElement = None
    for ix, grammarElement in enumerate(m1_grammar.split(' ')):
        terms = grammarElement.split(',')
        currOffset += float(terms[1]) # works just fine

        # Case 1: it's a rest. Just append
        if terms[0] == 'R':
            rNote = note.Rest(quarterLength = float(terms[1]))
            m1_elements.insert(currOffset, rNote)
            continue

        # Get the last chord first so you can find chord note, scale note, etc.
        try: 
            lastChord = [n for n in m1_chords if n.offset <= currOffset][-1]
        except IndexError:
            m1_chords[0].offset = 0.0
            lastChord = [n for n in m1_chords if n.offset <= currOffset][-1]

        # Case: no < > (should just be the first note) so generate from range
        # of lowest chord note to highest chord note (if not a chord note, else
        # just generate one of the actual chord notes). 

        # Case #1: if no < > to indicate next note range. Usually this lack of < >
        # is for the first note (no precedent), or for rests.
        if (len(terms) == 2): # Case 1: if no < >.
            insertNote = note.Note() # default is C

            # Case C: chord note.
            if terms[0] == 'C':
                insertNote = generate_chord_tone(lastChord)

            # Case S: scale note.
            elif terms[0] == 'S':
                insertNote = generate_scale_tone(lastChord)

            # Case A: approach note.
            # Handle both A and X notes here for now.
            else:
                insertNote = generate_approach_tone(lastChord)

            # Update the stream of generated notes
            insertNote.quarterLength = float(terms[1])
            if insertNote.octave < 4:
                insertNote.octave = 4
            m1_elements.insert(currOffset, insertNote)
            prevElement = insertNote

        # Case #2: if < > for the increment. Usually for notes after the first one.
        else:
            # Get lower, upper intervals and notes.
            interval1 = interval.Interval(terms[2].replace("<",''))
            interval2 = interval.Interval(terms[3].replace(">",''))
            if interval1.cents > interval2.cents:
                upperInterval, lowerInterval = interval1, interval2
            else:
                upperInterval, lowerInterval = interval2, interval1
            lowPitch = interval.transposePitch(prevElement.pitch, lowerInterval)
            highPitch = interval.transposePitch(prevElement.pitch, upperInterval)
            numNotes = int(highPitch.ps - lowPitch.ps + 1) # for range(s, e)

            # Case C: chord note, must be within increment (terms[2]).
            # First, transpose note with lowerInterval to get note that is
            # the lower bound. Then iterate over, and find valid notes. Then
            # choose randomly from those.
            
            if terms[0] == 'C':
                relevantChordTones = []
                for i in range(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if is_chord_tone(lastChord, currNote):
                        relevantChordTones.append(currNote)
                if len(relevantChordTones) > 1:
                    insertNote = random.choice([i for i in relevantChordTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantChordTones) == 1:
                    insertNote = relevantChordTones[0]
                else: # if no choices, set to prev element +-1 whole step
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                m1_elements.insert(currOffset, insertNote)

            # Case S: scale note, must be within increment.
            elif terms[0] == 'S':
                relevantScaleTones = []
                for i in range(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if is_scale_tone(lastChord, currNote):
                        relevantScaleTones.append(currNote)
                if len(relevantScaleTones) > 1:
                    insertNote = random.choice([i for i in relevantScaleTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantScaleTones) == 1:
                    insertNote = relevantScaleTones[0]
                else: # if no choices, set to prev element +-1 whole step
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                m1_elements.insert(currOffset, insertNote)

            # Case A: approach tone, must be within increment.
            # For now: handle both A and X cases.
            else:
                relevantApproachTones = []
                for i in range(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if is_approach_tone(lastChord, currNote):
                        relevantApproachTones.append(currNote)
                if len(relevantApproachTones) > 1:
                    insertNote = random.choice([i for i in relevantApproachTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantApproachTones) == 1:
                    insertNote = relevantApproachTones[0]
                else: # if no choices, set to prev element +-1 whole step
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                m1_elements.insert(currOffset, insertNote)

            # update the previous element.
            prevElement = insertNote

    return m1_elements


def is_chord_tone(lastChord, note):
    return (note.name in (p.name for p in lastChord.pitches))

''' Helper function to generate a chord tone. '''
def generate_chord_tone(lastChord):
    lastChordNoteNames = [p.nameWithOctave for p in lastChord.pitches]
    return note.Note(random.choice(lastChordNoteNames))

''' Helper function to generate a scale tone. '''
def generate_scale_tone(lastChord):
    # Derive major or minor scales (minor if 'other') based on the quality
    # of the lastChord.
    scaleType = scale.WeightedHexatonicBlues() # minor pentatonic
    if lastChord.quality == 'major':
        scaleType = scale.MajorScale()
    # Can change later to deriveAll() for flexibility. If so then use list
    # comprehension of form [x for a in b for x in a].
    scales = scaleType.derive(lastChord) # use deriveAll() later for flexibility
    allPitches = list(set([pitch for pitch in scales.getPitches()]))
    allNoteNames = [i.name for i in allPitches] # octaves don't matter

    # Return a note (no octave here) in a scale that matches the lastChord.
    sNoteName = random.choice(allNoteNames)
    lastChordSort = lastChord.sortAscending()
    sNoteOctave = random.choice([i.octave for i in lastChordSort.pitches])
    sNote = note.Note(("%s%s" % (sNoteName, sNoteOctave)))
    return sNote

''' Helper function to generate an approach tone. '''
def generate_approach_tone(lastChord):
    sNote = generate_scale_tone(lastChord)
    aNote = sNote.transpose(random.choice([1, -1]))
    return aNote

''' Helper function to generate a random tone. '''
def generate_arbitrary_tone(lastChord):
    return generate_scale_tone(lastChord) # fix later, make random note.

def prune_grammar(curr_grammar):
    pruned_grammar = curr_grammar.split(' ')

    for ix, gram in enumerate(pruned_grammar):
        terms = gram.split(',')
        terms[1] = str(round_up_down(float(terms[1]), 0.250, 
            random.choice([-1, 1])))
        pruned_grammar[ix] = ','.join(terms)
    pruned_grammar = ' '.join(pruned_grammar)

    return pruned_grammar


''' Remove repeated notes, and notes that are too close together. '''
def prune_notes(curr_notes):
    for n1, n2 in grouper(curr_notes, n=2):
        if n2 == None: # corner case: odd-length list
            continue
        if isinstance(n1, note.Note) and isinstance(n2, note.Note):
            if n1.nameWithOctave == n2.nameWithOctave:
                curr_notes.remove(n2)

    return curr_notes


''' Perform quality assurance on notes '''
def clean_up_notes(curr_notes):
    removeIxs = []
    for ix, m in enumerate(curr_notes):
        # QA1: ensure nothing is of 0 quarter note len, if so changes its len
        if (m.quarterLength == 0.0):
            m.quarterLength = 0.250
        # QA2: ensure no two melody notes have same offset, i.e. form a chord.
        # Sorted, so same offset would be consecutive notes.
        if (ix < (len(curr_notes) - 1)):
            if (m.offset == curr_notes[ix + 1].offset and
                isinstance(curr_notes[ix + 1], note.Note)):
                removeIxs.append((ix + 1))
    curr_notes = [i for ix, i in enumerate(curr_notes) if ix not in removeIxs]

    return curr_notes


def round_up_down(num, mult, upDown):
    if upDown < 0:
        return round_down(num, mult)
    else:
        return round_up(num, mult)


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def round_down(num, mult):
    return (float(num) - (float(num) % mult))


def round_up(num, mult):
    return round_down(num, mult) + mult
