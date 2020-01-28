# Header block
print("Lionel Dsilva")
print("DATA-51100")
print("Week 1 - Assignment 1")
print("Spring 2020")

#initial code execution block
try:
    num_Input = int(input("Enter a positive number: "))
    num = []
except ValueError:
    print('String was entered instead on integer')
except NameError:
    print('String is present, unable to continue')
# Loop
finally:
    while num_Input >= 0:
        num.append(num_Input)
        mean = sum(num) / len(num)
        var = sum((xi-mean)**2 for xi in num) / (len(num))
        if len(num) == 1:
            var_Sample = sum([(xi-mean)**2 for xi in num]) / (len(num))
        else:
            var_Sample = sum([(xi - mean)**2 for xi in num]) / (len(num) - 1)
        
        print("The mean is: ", mean)
        print("The variance (population) is: ", var)
        print("The sample variance is: ", var_Sample)
        
        try:
            num_Input = int(input("Enter a number: "))
        
        except ValueError:
            print("Character was entered")
            print("Program was terminated")
            break
        finally:
            if num_Input <0:
                print("Program has terminated")
                break
                
                try:
                    if num_Input == 'q':
                        print("Character was entered")
                        break
                except TypeError:
                    print("Type error was detected")
                    print("Program will now quit")
                    break
                except:
                    print("Seems like there was something else wrong")
                finally:
                    print("Program has terminated")
                    break
        

#list initialises but loops to infinite
#fixed infinite loop
#experimenting with try/except for all possible errors
#reached potential conclusion
#noted variance formula is off compared to statistics.variance() and numpy.var()
#variance formula fixed to sample variance formula 
#corrected zeroDivisionError
#Program works - assignment complete