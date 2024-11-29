# RAW DATA
- LMD-full-MIDI total of 7489 npy files, each file is of the following format:
    * ...
    * raw lyric
        * sentence 1
            * word 1
            * word 2
            * word 3
            * ...
        * sentence 2
        * sentence 3
        * ...
    * raw melody
        * sentence 1
            * word 1
                * syllable 1
                    * pitch
                    * duration_of_the_note
                    * duration_of_rest_before_the_note
                * syllable 2
                * syllable 3
                * ...
            * word 2
            * word 3
            * ...
        * sentence 2
        * sentence 3
        * ...
    * ...
- 1 word in a sentence in lyric can match to multiple syllables in melody

# Processed LMD data
- after running src/generate_lmd_data.py, each file is processed into the following format:
    * lyric: syllable, \[align\], syllable,\[align\], syllable, \[sep\], syllable ... 
    * melody: note_MIDI_NUM, dur_MIDI_NUM, note_MIDI_NUM, dur_MIDI_NUM, ... \[align\], note_MIDI_NUM, dur_MIDI_NUM, ... \[sep\] ...
- align is word-level alignment flag, sep is sentence level alignment flag
