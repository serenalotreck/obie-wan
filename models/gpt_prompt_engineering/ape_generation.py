"""
Use the Automatic Prompt Engineer (APE) from the paper  "Large Language Models
Are Human-Level Prompt Engineers". The APE repo must be installed outside this
repo and installed into the venv with the command pip install -e .

Author: Serena G. Lotreck
"""
from automatic_prompt_engineer import ape

sentences = [
        ("Analysis of Arabidopsis plants containing the HDA19:beta-glucuronidase "
        "fusion gene revealed that HDA19 was expressed throughout the life of "
        "the plant and in most plant organs examined."),
        ("Overexpression of HDA19 in 35S:HDA19 plants decreased histone "
        "acetylation levels, whereas downregulation of HDA19 in HDA19-RNA "
        "interference (RNAi) plants increased histone acetylation levels."),
        ("Jasmonates play a central signaling role in damage-induced nicotine "
        "formation."),
        ("Transcriptional regulation by ERF and cooperatively acting MYC2 "
        "transcription factors are corroborated by the frequent occurrence of "
        "cognate cis-regulatory elements of the factors in the promoter regions "
        "of the downstream structural genes."),
        ("A number of molecular species can function as signal molecules or "
        "elicitors of phytoalexin synthesis, including poly- and "
        "oligosaccharides, proteins and polypeptides, and fatty acids."),
        ("It is estimated that about 50% of the world rice production is "
        "affected mainly by drought."),
        ("Considering differentially expressed genes in the co-expressed "
        "modules and supplementing external information such as "
        "resistance/tolerance QTLs, transcription factors, network-based "
        "topological measures, we identify and prioritize drought-adaptive "
        "co-expressed gene modules and potential candidate genes."),
        ("Antifungal activity of these amides was shown by inhibition of "
        "conidial germination and germ tube elongation of F. graminearum and "
        "Alternaria brassicicola, indicating that they act as phytoalexins."),
        ("Thus, the induced accumulation of two groups of phenylamides, "
        "cinnamic acid amides with indole amines, and p-coumaric acid amides "
        "with putrescine and agmatine related amines, represents a major "
        "metabolic response of wheat to pathogen infection.")
        ]

triples = [
        ('['
            '("HDA19:beta-glucuronidase fusion gene", "is-in", "Arabidopsis"), '
            '("HDA19", "is-in", "Arabidopsis")'
            ']'),
        ('['
            '("HDA19", "is-in", "35S:HDA19 plants"), '
            '("HDA19", "inhibits", "histone acetylation"), ' # NOTE TO SELF: This
                                                        # is fixed here but
                                                        # needs to be fixed in
                                                        # the PICKLE corpus,
                                                        # incorrect relation
                                                        # due to prepositional
                                                        # phrase
            '("HDA19", "is-in", "HDA19-RNA interference (RNAi) plants")'
            ']'),
        ('['
            '("Jasmonates", "interacts", "damage-induced nicotine formation")'
            ']'),
        ('['
            '("ERF", "interacts", "MYC2 transcription factors")'
            ']'),
        ('['
            '("fatty acids", "activates", "phytoalexin synthesis"), '
            '("polypeptides", "activates", "phytoalexin synthesis"), '
            '("proteins", "activates", "phytoalexin synthesis"), '
            '("poly- and oligosaccharides", "activates", "phytoalexin synthesis")'
            ']'),
        ('['
            '"No relation"'
            ']'),
        ('['
            '"No relation"'
            ']'),
        ('['
            '("amides", "inhibits", "conidial germination"), '
            '("amides", "inhibits", "germ tube elongation"), '
            '("amides", "inhibits", "F. graminearum"), '
            '("amides", "inhibits", "Alternaria brassicicola")'
            ']'),
        ('['
            '("wheat", "produces", "putrescine and agmatine related amides"), '
            '("wheat", "produces", "p-coumaric acid amides"), '
            '("wheat", "produces", "indole amides"), '
            '("wheat", "produces", "cinnamic acid amides"), '
            '("wheat", "produces", "phenylamides")'
            ']')
        ]

# Copied from example, not sure if this needs to be different
eval_template = \
        """Instruction: [PROMPT]
        Input: [INPUT]
        Output: [OUTPUT]"""

simple_result, simple_demo_fn = ape.simple_ape(
        dataset=(sentences, triples),
        eval_template=eval_template,
        num_prompts=20,
        eval_rounds=100
        )


print(f'\nSimple APE Result:\n{simple_result}')

manual_prompt = ('Extract ("Subject", "Predicate", "Object") triples from '
                'the following sentence, choosing the predicate from the list '
                '[activates, inhibits, produces, interacts, is-in] and '
                'excluding prepositions and specifiers:')

human_result = ape.simple_eval(
            dataset=(sentences, triples),
            eval_template=eval_template,
            prompts=[manual_prompt],
                    )

print(f'\nHuman result:\n{human_result}')
