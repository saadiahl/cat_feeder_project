'''
contributors: saadiahlazim4@gmail.com, Esther Kang
'''
import torch

class Cat:
 
    def __init__(self, name: str, activity_level: str, weight: float):
        self.name: str = name
        self.weight: float = weight
        self.activity_level: str = activity_level  # Keep this as a string
        
        self.activity: float = self.set_activity_level(activity_level)
        self.weight_level: str = self.weightLevel()
        self.meals_per_day: int = self.numOfMeals()
        self.age: int = int(input(f'How old is {name}? (If younger than 1, enter 0.) '))
        # self.age = self.getAge(name, weight, self.activity, self.meals_per_day)
        
        self.portion: float = self.calculate_portion()
        self.suggestedPortions: list[float] = [self.portion] * self.meals_per_day

        # new
        self.suggestedDailyPortions: float = sum(self.suggestedPortions)
        self.eatenPortions: list[float] = []
        self.sessionNum: int = 0
        self.nextSessionPortion: float = self.suggestedPortions[0]
        self.entering: bool = True
        self.eating: bool = False

    def calculate_portion(self):
        if self.age < 1:
            return self.forKitten(self.name, self.weight)
        elif self.age >= 11:
            return self.forSenior(self.name, self.weight, self.activity, self.meals_per_day)
        else:
            if self.weight_level == 'underweight':
                return self.forUnderWeight(self.weight, self.activity, self.meals_per_day)
            elif self.weight_level == 'overweight':
                return self.forOverWeight(self.weight, self.activity, self.meals_per_day)
            else:  # ideal weight
                return self.forIdealWeight(self.weight, self.activity, self.meals_per_day)

    def set_activity_level(self, activity_level: str) -> float:
        if activity_level == 'Inactive':
            activity = 1.2
        elif activity_level == 'Average':
            activity = 1.4
        elif activity_level == 'High':
            activity = 1.6
        else:
            print('Invalid input. Using default activity level (Average).')
            activity = 1.4
        return float(activity)

    def forIdealWeight(self, weight, activity, meals_per_day) -> float:
        requirement = (weight**0.75 * 70 * activity)
        portion = requirement / meals_per_day
        return portion

    def forUnderWeight(self, weight, activity, meals_per_day):
        portion = self.forIdealWeight(weight, activity, meals_per_day)
        final_portion = int(portion) * 1.2
        return final_portion

    def forOverWeight(self, weight, activity, meals_per_day):
        portion = self.forIdealWeight(weight, activity, meals_per_day)
        final_portion = int(portion) * 0.9
        return final_portion

    def forKitten(self, name, weight):
        months = int(input(f'How many months is {name}? '))
        if months < 4:
            portion = self.forUnderWeight(weight, 1.6, 3)
        else:
            portion = self.forUnderWeight(weight, 1.2, 3)
        return portion

    def forSenior(self, name, weight, activity, meals_per_day):
        portion = self.forIdealWeight(weight, activity, meals_per_day)
        return portion

    def forAdult(self, weight, activity, meals_per_day):
        portion = self.forIdealWeight(weight, activity, meals_per_day)
        return portion
    
    # def getAge(self, name, weight, activity, meals_per_day):
    #     age = int(input(f'How old is {name}? (If younger than 1, enter 0.) '))
    #     if age < 1:
    #         portion = self.forKitten(name, weight)
    #     elif age >= 11:
    #         portion = self.forSenior(name, weight, activity, meals_per_day)
    #     else:
    #         portion = self.forAdult(weight, activity, meals_per_day)
    #     return portion
    

    def weightLevel(self) -> str:
        q1 = input('Can you feel an abdominal fat pad? (Y/N) ')
        if q1 == 'Y':
            q2 = input('Does your cat have an abdominal tuck and an obvious waist? (Y/N) ')
            if q2 == 'Y':
                weight_level = 'ideal'
            else:
                q3 = input('Are the ribs difficult to feel? (Y/N) ')
                weight_level = 'overweight' if q3 == 'Y' else 'ideal'
        elif q1 == 'N':
            q2_2 = input("Can you see your cat's ribs? (Y/N) ")
            weight_level = 'underweight' if q2_2 == 'Y' else 'ideal'
        else:
            print('Invalid input. Assuming ideal weight.')
            weight_level = 'ideal'
        return weight_level
    
    def getSessionNum(self):
        return self.sessionNum

    def setSessionNum(self):
        self.sessionNum+=1
        return
    
    def getSession(self) -> bool:
        if self.sessionNum<=self.meals_per_day: 
            return True
        elif self.nextSessionPortion > 0:
            self.meals_per_day+=1
            return True
        else:
            return False
    
    def setSession(self): # Overwrite
        return

    def getSessionPortion(self):
        return self.nextSessionPortion

    def setSessionPortion(self, eatenWeight: float):
        self.setSessionNum()
        self.eatenPortions.append(eatenWeight)
        self.nextSessionPortion = (self.suggestedDailyPortions-self.eatenPortions[self.sessionNum-1])/(self.meals_per_day-self.sessionNum)
        return

    def numOfMeals(self) -> int:
        if self.weight_level == 'underweight':
            if self.activity_level == 'High':
                return 5
            elif self.activity_level == 'Average':
                return 3
            else :
                return 2
        elif self.weight_level == 'ideal':
            if self.activity_level == 'High':
                return 4
            else :
                return 3
        else :
            if self.activity_level == 'High':
                return 3
            else :
                return 2

    def __str__(self):
        return (f"Name: {self.name}\nWeight Level: {self.weight_level}\n"
                f"Activity Level: {self.activity_level}\nNumber of eating sessions: {self.meals_per_day}\n"
                f"Grams per meal: {self.portion:.2f}\nTotal daily portion: {sum(self.suggestedPortions):.2f}")

# cats = {}
# while True:
#     name = input("Enter cat's name (or 'done' to finish): ")
#     if name.lower() == 'done':
#         break
#     weight = float(input(f'What is the weight of {name}? In kilograms. '))
#     activity_level = input(f'Which activity level does your cat belong to? (Inactive/Average/High) ')
#     cat = Cat(name, activity_level, weight)
#     cats[name] = cat

# for cat in cats:
#     print(cat)

# torch.save(cats.copy(), "./cats.pt")