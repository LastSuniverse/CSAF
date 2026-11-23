"""Text anchor sets used in the CSAF paper (verbatim, as recorded).

Domain anchors are hand-written per dataset; UNI6 is the dataset-
neutral control. The published UCF configuration uses
UCF_ANOMALOUS[1:] (7 entries here; the first is the generic
singleton used only in the single-anchor ablation). Audio anchors
feed the CLAP branch.
"""

UCF_ANOMALOUS = [
    'an anomalous or criminal event in surveillance footage',
    'a dangerous or illegal incident captured by a security camera',
    'suspicious criminal activity happening in a surveillance video',
    'an unusual event that poses a threat to safety or property',
    'a crime or violent act in progress in the scene',
    'abnormal behavior indicating an emergency or an offense',
    'a security threat or law-breaking event in progress',
]

UCF_NORMAL = [
    'people walking normally in a surveillance scene',
    'an ordinary surveillance scene with no unusual events',
    'normal everyday activity captured by a security camera',
    'a peaceful public scene with regular pedestrian activity',
    'routine daily activity with nothing unusual happening',
    'a calm street or indoor view with people going about their business',
]

XD_ANOMALOUS = [
    'a violent or dangerous event in a video',
    'physically aggressive behavior causing harm',
    'a fight, assault or act of violence in progress',
    'an explosion, shooting or weapon attack scene',
    'a violent scene with blood, injury or destruction',
    'people being attacked, beaten or threatened',
]

XD_NORMAL = [
    'a peaceful scene with calm everyday activity',
    'people talking, walking or working without aggression',
    'an ordinary scene from daily life or a movie with no violence',
    'a calm conversation or quiet moment in a video',
    'normal social interaction with no physical conflict',
    'a tranquil indoor or outdoor scene with nothing unusual',
]

UNI_ANOMALOUS = [
    'an anomalous event occurring in the video',
    'an unusual incident that deviates from normal activity',
    'abnormal or dangerous behavior happening in the scene',
    'something irregular that breaks the routine of the scene',
    'an unexpected event posing a risk to people or property',
    'an abnormal situation unfolding in front of the camera',
]

UNI_NORMAL = [
    'normal everyday activity in the scene',
    'a routine moment with nothing unusual happening',
    'ordinary calm behavior continuing as expected',
    'a typical scene following its usual course',
    'regular activity with no sign of trouble',
    'a peaceful ordinary moment in the video',
]

AUDIO_ANOMALOUS = [
    'gunshots and gunfire',
    'a car crash with screeching tires and impact',
    'screaming and shouting in panic',
    'an explosion blast',
    'breaking glass and violent smashing',
    'loud alarms, sirens and chaotic noise',
]

AUDIO_NORMAL = [
    'calm ambient sounds',
    'quiet conversation and gentle background noise',
    'peaceful everyday sounds',
    'soft background music',
]

INSTRUCTION = "Describe the surveillance scene and the behavior of people in it."
