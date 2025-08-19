(
s.waitForBoot {
    ~basePath = \"../songs\".standardizePath;
    ~playlist = [
        (~basePath + \"/Miles_Hollifield_KillEmWithTheBeat.wav\").standardizePath
    ];
    ~trackIndex = 0;

    ~rateBus = Bus.control(s, 1);
    ~pitchBus = Bus.control(s, 1);
    ~ampBus = Bus.control(s, 1);

    ~loadTrack = {
        |index|
        if(index >= 0 and: { index < ~playlist.size }) {
            ~buffer = Buffer.read(s, ~playlist[index]);
        };
    };

    ~player = {
        SynthDef(\HandDJPlayer, {
            |out=0|
            var sig, rate, amp, freq, smoothedRate, smoothedAmp, smoothedFreq;
            
            rate = In.kr(~rateBus);
            amp = In.kr(~ampBus);
            freq = In.kr(~pitchBus);

            smoothedRate = Lag.kr(rate, 0.1); // SigTo-like smoothing
            smoothedAmp = Lag.kr(amp, 0.1);
            smoothedFreq = Lag.kr(freq, 0.1);

            sig = PlayBuf.ar(1, ~buffer, rate: smoothedRate, doneAction: 2);
            sig = PitchShift.ar(sig, 0.2, smoothedFreq / 440, 0, 0.01, 0.01);
            sig = sig * smoothedAmp;
            Out.ar(out, sig.dup);
        }).add;
    };

    ~spawnPlayer = {
        Synth(\HandDJPlayer);
    };

    ~loadTrack.(~trackIndex);
    ~player.();

    OSCdef.new(
        \hand_dj,
        { |msg|
            var cmd = msg[1];
            switch(cmd,
                "vol", { ~ampBus.set(msg[2].clip(0.0, 10.0) / 10.0); },
                "rate", { ~rateBus.set(msg[2].clip(0.1, 2.0)); },
                "freq", { ~pitchBus.set(msg[2].clip(20, 600)); },
                "play", { ~spawnPlayer.(); },
                "next", {
                    ~trackIndex = (~trackIndex + 1) % ~playlist.size;
                    ~loadTrack.(~trackIndex);
                },
                "prev", {
                    ~trackIndex = (~trackIndex - 1);
                    if (~trackIndex < 0) { ~trackIndex = ~playlist.size - 1; };
                    ~loadTrack.(~trackIndex);
                },
                "reset", {
                    ~rateBus.set(1.0);
                    ~pitchBus.set(440);
                    ~ampBus.set(0.5);
                }
            );
        },
        '/hand_dj'
    );

    ~rateBus.set(1.0);
    ~pitchBus.set(440);
    ~ampBus.set(0.5);

    "Hand DJ server with SigTo-like smoothing ready.".postln;
};
)
