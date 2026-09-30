"""Fixed inputs for the eval; "caption" stands in for the vision model's description of an attachment."""

CASES = [
    {"id": "krea2/simple", "function": "krea2", "text": "a lighthouse on a cliff"},
    {"id": "krea2/anime", "function": "krea2", "text": "anime style girl waiting at a rural bus stop in summer"},
    {"id": "krea2/sign-text", "function": "krea2", "text": "a tiny ramen shop at night with a neon sign that says NOODLE BAR"},
    {"id": "krea2/non-english", "function": "krea2", "text": "un vieux pêcheur qui répare ses filets au lever du soleil"},
    {
        "id": "krea2/caption-only", "function": "krea2",
        "caption": "A golden retriever lies on a worn leather sofa beside a rain-streaked window. Soft grey daylight, "
        "a knitted blanket, a mug on a side table, muted warm tones.",
    },
    {
        "id": "krea2/intent-and-caption", "function": "krea2", "text": "same scene but as a watercolor, at sunset",
        "caption": "A red tram crosses a stone bridge over a river with old townhouses behind it. Overcast "
        "midday light, wide shot from the riverbank.",
    },
    {"id": "minimax/t2va-simple", "function": "minimax", "mode": "t2va", "text": "a cat knocks a glass off a table"},
    {
        "id": "minimax/t2va-dialogue", "function": "minimax", "mode": "t2va",
        "text": "[8s] two detectives argue in a rainy alley, one says we're out of time",
    },
    {
        "id": "minimax/t2va-clip-reference", "function": "minimax", "mode": "t2va", "duration": 5.0,
        "caption": "A skateboarder rolls down an empty concrete ramp at dusk, crouches, and ollies over a curb "
        "as the camera follows from behind.",
    },
    {
        "id": "minimax/i2va", "function": "minimax", "mode": "i2va", "text": "the kettle starts to boil",
        "caption": "A dented copper kettle sits on a gas stove in a small tiled kitchen, morning light through a "
        "window, steam not yet visible, live-action.",
    },
    {
        "id": "minimax/fl2va", "function": "minimax", "mode": "fl2va", "duration": 8.0,
        "caption": "Opening: a paper boat rests on a puddle beside a curb. Ending: the same boat drifts into a "
        "storm drain. Overcast afternoon, wet asphalt, live-action.",
    },
    {
        "id": "minimax/l2va", "function": "minimax", "mode": "l2va", "text": "[6s] the band's final chord",
        "caption": "A guitarist in a leather jacket stands on a smoky stage with arms raised, lit from behind by "
        "a white spotlight, crowd hands silhouetted in the foreground.",
    },
]
