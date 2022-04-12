## DVD Rental CO - Case Study

The marketing team of the Rental company has given us a few requirements to meet, so as to send out personalized recommendations via email to their customers; based on either genre or
subject matter.

### Requirements:

The requirements are as follows:
- For each customer, identify top 2 categories based on their past rental history. This requirement will drive marketing creative images, as seen in the draft email.
- The marketing team requests for the 3 most popular films for each customer's top 2 categories. However, we cannot recommend a film that has already been viewed by the customer.
Customers that do not have any film recommendations for either category must be flagged out so marketing can exclude them from the email campaign.
- The number of films watched by each customer in their top 2 categories is required, as well as some specific insights:
  - We will need how many total films a customer has watched in their top category.
  - How many more films has the customer watched compared to the average rental customer.
  - How does the customer rank in terms of the top X% compared to all other customers in this film category.
- There is a secondary ranking category as well:
  - How many total films has the customer watched in a specific category.
  - What proportion of each customer's total films watched does this count make.
- Alongside the top two categories, marketing has also requested top actor film recommendations where up to 3 more films are included in the recommendations list as well as
the count of films by the top actor. Actors are chosen in alphabetical order in the even of a count tie. In the event that a customer does not have at least one recommendation,
they must be flagged with a separate actor exclusion flag.
