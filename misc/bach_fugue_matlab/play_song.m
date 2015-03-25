function xx = play_song()

song = load('bach_fugue.mat');
theVoices = song.theVoices;

fs = 44100;
bpm = 120;
beats_per_second = bpm/60;
seconds_per_beat = 1 / beats_per_second;
seconds_per_pulse = seconds_per_beat/4;

longestVoice = 1;

for ii = 1:length(theVoices)
    longestVoice(theVoices(longestVoice).startPulses(end) <= theVoices(ii).startPulses(end)) = ii;
end
disp('the longest voice is');
disp(ii);

xx = zeros(1, floor(theVoices(longestVoice).startPulses(end) * seconds_per_pulse * fs)+fs);

for idx = 1:length(theVoices)
    if length(theVoices(idx).startPulses) < 1
       continue; 
    end
    %xx = zeros(1, floor(theVoices(idx).startPulses(end) * seconds_per_pulse * fs)+fs);
    for jdx = 1:length(theVoices(idx).startPulses)
       startTime = floor(theVoices(idx).startPulses(jdx) * seconds_per_pulse * fs);
       endTime = startTime + floor(theVoices(idx).durations(jdx) * seconds_per_pulse * fs);
       keynum = theVoices(idx).noteNumbers(jdx);
       tone = key2note(0.1, keynum, theVoices(idx).durations(jdx)*seconds_per_pulse);
       xx(startTime:endTime) = xx(startTime:endTime) + tone;
    end
end

soundsc(xx, 44100);
end