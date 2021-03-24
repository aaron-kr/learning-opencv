# testing.py
# 6. TESTING - Test our classifier
import random

# Load test data
TEST_LIST = helpers.load_dataset(imgs_test)

# Standardize the data
STD_TEST_LIST = helpers.standardize(TEST_LIST)

# Shuffle the standardized test data
random.shuffle(STD_TEST_LIST)

# 7. ACCURACY - Determine the accuracy
def get_misclassified_imgs(test_imgs):
  # Place misclassified imgs in a list
  misclassified_img_lbls = []

  # Iterate through all test imgs
  # Classify each and compare to the true label
  for im in test_imgs:
    # Get true data
    i = im[0]
    true_lbl = im[1]

    # Get predicted label from classifier
    predicted_lbl = estimate_lbl(im)

    # Compare true and predicted labels
    if ( predicted_lbl != true_lbl ):
      # If not the same, the img was misclassified
      misclassified_img_lbls.append((i, predicted_lbl, true_lbl))
  
  # Return misclassified list [image, predicted_lbl, true_lbl]
  return misclassified_img_lbls

# 8. ACCURACY RATE - an important metric
# Find all misclassified imgs in a given test set
MISCLASSIFIED = get_misclassified_imgs(STD_TEST_LIST)

# Accuracy calculations
total = len(STD_TEST_LIST)
correct = total - len(MISCLASSIFIED)
accuracy = correct / total

print('Accuracy: ', str(accuracy))
print('Number misclassified: ' + str(len(MISCLASSIFIED)) + ' out of ' + str(total))

# 9. CORRECTION - now, visualize the misclassified imgs, and see how you can correct your algorithm
# Visualize misclassified example(s)
num = 0
test_mis_im = MISCLASSIFIED[num][0]
test_mis_lb = MISCLASSIFIED[num][1]

# Display an image in the list
plt.imshow(test_mis_im)

# Print its predicted label
print("Classified as: ", test_mis_lb)