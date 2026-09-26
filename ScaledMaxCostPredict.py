#for plotting the data points
import matplotlib.pyplot as plot

#the data gathered from cars.com
years = [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 
2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000, 1999, 1998, 1997, 1996, 1995, 1994, 1993, 1992]

Maxprice = [57270, 56070, 55720, 40945, 39730, 39035, 38675, 38565,
26070, 25800, 25805, 24350, 24425, 26670, 26795, 26015, 25450, 25010, 24335, 19785, 19435, 19695, 20325, 20295, 19571, 18328, 16535, 14840]


m = len(years)
#starting with small weights for baseline
weight0 = .01
weight1 = 0.5

#changed the learning rate from 0.0001 to 0.000000000000001 because of overflow
#since that learning rate was for min I had to change it for max since max has a larger number than min prices so it got changed to 0.0000000000000001
LearningRate = 0.001 #played around with the number until it's around the middle between the two yrs
decayRate = 0.95 #learning rate will reduce by 5% every loop


#scaling the year
MeanYr = sum(years)/m
StdDevYr = (sum((x-MeanYr)**2 for x in years)/m) **0.5
ScaleYr = [(x-MeanYr)/StdDevYr for x in years]

#scaling the max price 
MeanMax = sum(Maxprice)/m
StdDevMax = (sum((y-MeanMax)**2 for y in Maxprice) /m) **0.5
ScaleMaxPrice = [(y-MeanMax)/StdDevMax for y in Maxprice]

CostHistory = [] #To store the cost for plotting later
for i in range(100):
    totalCost = 0
    gradient0 = 0
    gradient1 = 0

    #loops through every data point
    for j in range(m): #this is the m above the sigma
        x = ScaleYr[j]
        y = ScaleMaxPrice[j]

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

    LearningRate = LearningRate * decayRate

print(f"Final Weights: theta0 = {weight0}, theta1 = {weight1}")

MissingYear = [2012,2013, 2014, 2015, 2016, 2017, 2018]
PredictPrice = []

#using the weight found I used it to predict the max cost for 2012-2018 and rounded to 2 decimal places
for year in MissingYear:
    #scaling the missing years so it's not big number and now small numbers
    scaledMissingYr = (year-MeanYr)/StdDevYr

    Scaledprediction = weight0 + (weight1*scaledMissingYr)

    RealPrediction = (Scaledprediction*StdDevMax) + MeanMax

    PredictPrice.append(RealPrediction)
    print(f"Predicted Scaled Maximum Price for {year}: ${RealPrediction:.2f}")

iterations = range(1, 101)

plot.figure(figsize=(10,10)) #creates the canvas size
plot.plot(iterations, CostHistory, color="red", linewidth=2) #goes through 1-100 and puts the CostHistory from each iternation
plot.title("Scaled Maximum Price Loss Curve ")
plot.xlabel("Iterations")
plot.ylabel("Cost (J)")
plot.grid(True) #have a grid background
plot.savefig("ScaledMaxPrice_LossCurve.png")


plot.figure(figsize=(10,6))
#Plot the data from the website
plot.scatter(years, Maxprice, color="purple", label="Acutal Max Price" )

#Plot the missing data from 2012-2018
plot.scatter(MissingYear, PredictPrice, color="red", marker="x", s=100, label="Predicted Max Price")
plot.title("Scaled Actual vs Predicted Max Prices")
plot.xlabel("Years")
plot.ylabel("Price ($)")
plot.legend()
plot.grid(True)
plot.savefig("ScaledYeartoMaxPrice.png")