# get customers with at least one interaction from dataset
# for those that the customer did not interact with, population the score as 0
#  train the model the customer interactions. (with 0 and 1), based on customers with at least one interaction
### The model may return customers who are 'nan' if you set coldstartstrategy = 'nan' (for you to do your own coldstart),
# if you set it to 'drop' then no predictions will be made for those customers
# predict the model based on the crossJoin (cin, service) of customers that appeared during your evaluation period
# evaluate the predictions (filter top 3 per customer) then compare against the actual (cin, service) interactions
# Get the hit rate