# Protocol v1.2: required five-seed E1 evaluation

This revision supersedes the single-seed E1 rule in [protocol v1.1](protocol-v1.1.md). All other locked choices, including F0, chronological folds, training and evaluation stock universes, baseline preprocessing, official score, and holdout isolation, remain in force. The [team workflow](README.md) and [agent guide](../AGENTS.md) describe the current procedure.

For every E1 model configuration, run each of the three expanding validation folds with seeds **42, 64, 128, 256, and 512**. Seed 42 remains the reference seed. Within a seed, average the three yearly official scores equally. Average the five seed-level CV scores equally to compare configurations. Record all 15 fold scores, the five seed-level CV scores, their mean, and population standard deviation. Do not pick the best seed.

The five runs measure sensitivity to model randomness. A deterministic configuration may give identical predictions for all seeds, especially when its training procedure does not sample rows or features. Identical scores must be reported honestly. Repeating seeds does not remove uncertainty from the three validation years or guarantee a more robust final model. Model-specific sampling and optimization choices remain under the approach owner's control, with the E1 configuration recorded.

Which seed or ensemble, if any, to use for final deployment refitting remains undecided.
