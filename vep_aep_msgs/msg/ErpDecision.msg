# One evoked-potential decision published by erp_detector
std_msgs/Header header        # stamp = time of decision
int64 onset_sample            # absolute sample index of the stimulus onset
int32 true_code               # stimulus code from the marker stream (1 LA, 2 RA, 3 LV, 4 RV)
int32 predicted               # predicted class (modality: 0 AEP / 1 VEP; 4-class: 1..4)
string label                  # human-readable label, e.g. "VEP-left"
float64 score                 # classifier decision value
builtin_interfaces/Time onset_stamp   # wall time at which the onset sample was acquired
float64 classify_us           # classifier compute time
