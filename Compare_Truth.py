import sys

# reads the two files as command line argument
# Example: Comare_Truth.py <trainset> <testset>
def read_files():
    with open(sys.argv[1], 'r', encoding='utf-8') as train:
        truthData = train.readlines()   # copy the content of the file in a list

    with open(sys.argv[2], 'r', encoding='utf-8') as test:
        goldData = test.readlines()

    return truthData, goldData


def copare_truth_with_gold(truthData, goldData):

    gen_match=0
    age_match=0

    for truth in truthData:
        truth = truth.split(":::")
        author_id = truth[0]
        gender = truth[1]
        age = truth[2].split("\n")
        age = age[0]

        for gold in goldData:
            gold = gold.split(":::")

            if author_id == gold[0]:
                if gender == gold[1]:
                    gen_match+=1
                if age == gold[2]:
                    age_match+=1
                break

    accu_gen = gen_match/len(truthData)
    aucu_age = age_match/len(truthData)

    print("\nAccuracy of Gender = %.2f" %accu_gen)
    print("\nAccuracy of Age = %.2f" %aucu_age)

    print("\nAccuracy (Average) = %.2f" %((accu_gen+aucu_age)/2.0))


def main():
    truthData, goldData = read_files()

    copare_truth_with_gold(truthData, goldData)


if __name__ == '__main__':
    main()
