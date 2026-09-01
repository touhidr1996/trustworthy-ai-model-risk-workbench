# Data card

## Provenance

Every record is generated locally by `trustbench.data.generate_applications`. No bank, borrower, customer, employee, credit-bureau, or government data is used. The generator and generated sample are MIT-licensed with this repository.

## Structure and assumptions

The dataset contains 12,000 fictional applications over 24 simulated months. Features describe synthetic income, debt ratio, history length, late payments, requested amount, and employment duration. `group_a` and `group_b` are abstract audit labels—not race, sex, nationality, age, or any real demographic category. Default outcomes come from a documented stochastic mechanism.

## Appropriate use

Suitable for software testing, teaching, and portfolio demonstration. It must not be used to estimate population behavior, train a real credit model, rank people, or justify lending policy.
