# Sweep of the exploratory PID generator over the full design: 8 indenters x 5 durations x 5 peak forces (1-5 N) x 5 profiles.
# Run from data_generation/exploratory/ (needs conf/ and objects/ as described in docs/dataset_preparation.md).
# In the thesis annexure the generator is called run_force_experiment.py; here it is demo_pybullet_digit_force.py.
$SCRIPT = "demo_pybullet_digit_force.py"

$INDENTERS = @("asterisk", "ball_array", "hash", "heart", "hexagonal", "ring", "star", "triangle")
$DURATIONS = @(10.0, 15.0, 20.0, 25.0, 30.0)
$FORCES = @(1, 2, 3, 4, 5)
$WAVEFORMS = @("triangular", "sinusoidal", "square", "trapezoidal", "sigmoid")

foreach ($INDENTER in $INDENTERS) {
    foreach ($DURATION in $DURATIONS) {
        foreach ($FORCE in $FORCES) {
            foreach ($WAVEFORM in $WAVEFORMS) {
                Write-Host "======================================================================"
                Write-Host "Running: Indenter=$INDENTER | Force=${FORCE}N | Time=${DURATION}s | Profile=$WAVEFORM"
                Write-Host "======================================================================"
                
                python $SCRIPT experiment.indenter_name=$INDENTER object=$INDENTER experiment.waveform=$WAVEFORM experiment.force_max_N=$FORCE experiment.duration_s=$DURATION
            }
        }
    }
}
Write-Host "All experimental combinations are complete."
