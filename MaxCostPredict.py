#for plotting the data points
import matplotlib.pyplot as plot

#the data gathered from cars.com
years = [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 
2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000, 1999, 1998, 1997, 1996, 1995, 1994, 1993, 1992]

Maxprice = [57270, 56070, 55720, 40945, 39730, 39035, 38675, 38565,
26070, 25800, 25805, 24350, 24425,26670, 26795, 26015, 25450, 25010, 24335, 19785, 19435, 19695, 20325, 20295, 19571, 18328, 16535, 14840]

m = len(years)
#starting with small weights for baseline
weight0 = .01
weight1 = 0.5

#changed the learning rate from 0.0001 to 0.000000000000001 because of overflow
#since that learning rate was for min I had to change it for max since max has a larger number than min prices so it got changed to 0.0000000000000001
LearningRate = 0.0000000000000001

CostHistory = [] #To store the cost for plotting later
for i in range(100):
    totalCost = 0
    gradient0 = 0
    gradient1 = 0

    #loops through every data point
    for j in range(m): #this is the m above the sigma
        x = years [j]
        y = Maxprice[j]

        h=weight0 + (weight1*x)

        #error rate calulate
        e = h-y

        #finding the cost J = (e^2) + 0.1 (e^4)
        cost = (e**2) +0.1 * (e**4)
        totalCost += cost

        #calculate gradients using the deriverative formulas
        gradient0 += (2*e)+0.4*(e**3)
        gradient1 += ((2*e)+0.4*(e**3))*x

    AvgCost = totalCost/m
    CostHistory.append(AvgCost)

    avgGradient0 = gradient0/m
    avgGradient1 = gradient1/m

    #updating the weights as needed
    weight0 = weight0 - (LearningRate * avgGradient0)
    weight1 = weight1 - (LearningRate * avgGradient1)

print(f"Final Weights: theta0 = {weight0}, theta1 = {weight1}")

MissingYear = [2012,2013, 2014, 2015, 2016, 2017, 2018]
PredictPrice = []

#using the weight found I used it to predict the max cost for 2012-2018 and rounded to 2 decimal places
for year in MissingYear:
    prediction = weight0 + (weight1*year)
    PredictPrice.append(prediction)
    print(f"Predicted maximum price for {year}: ${prediction:.2f}")

iterations = range(1, 101)

plot.figure(figsize=(10,10)) #creates the canvas size
plot.plot(iterations, CostHistory, color="red", linewidth=2) #goes through 1-100 and puts the CostHistory from each iternation
plot.title("Maximum Price Loss Curve")
plot.xlabel("Iterations")
plot.ylabel("Cost (J)")
plot.grid(True) #have a grid background
plot.savefig("MaxPrice_LossCurve.png")


plot.figure(figsize=(10,6))
#Plot the data from the website
plot.scatter(years, Maxprice, color="purple", label="Acutal Max Price" )

#Plot the missing data from 2012-2018
plot.scatter(MissingYear, PredictPrice, color="red", marker="x", s=100, label="Predicted Max Price")
plot.title("Actual vs Predicted Max Prices")
plot.xlabel("Years")
plot.ylabel("Price ($)")
plot.legend()
plot.grid(True)
plot.savefig("YeartoMaxPrice.png")