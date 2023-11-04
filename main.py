import cv2
import os
import streamlit as st
import numpy as np

def main():

    st.title("Fingerprint Matching App")
    st.header("ML Mini Project")
    st.subheader("Akshay Dongare (BC218)")
    st.subheader("Jaydatta Patwe (BC223)")
    st.subheader("Yash Dagadkhair (BC212)")

    input_filename = st.file_uploader("Upload an image for fingerprint matching", type=["jpg", "png", "bmp"])
    max_images_to_search = st.slider("Maximum Number of Images to Search", 0, 6000, 2000) #since ./real folder has 6k images 

    #input_filename = "SOCOFing/Altered/Altered-Hard/2__F_Left_thumb_finger_CR.BMP"
    #sample = cv2.imread("SOCOFing/Altered/Altered-Easy/140__F_Left_index_finger_CR.BMP")

    if input_filename is not None:

        sample = cv2.imdecode(np.fromstring(input_filename.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        st.image(sample, caption="Uploaded Image", use_column_width=True)

            
        #sample = cv2.imread(input_filename)

        best_score = 0
        filename = None
        image = None

        kp1, kp2, mp = None, None, None

        counter = 0
        progress_bar = st.progress(0)
        st.write("Looking through fingerprint database for the best match...")
        for file in [file for file in os.listdir("SOCOFing/Real")][:max_images_to_search]:
            if counter % 10 == 0:
                print("\nImage Number: ",counter)
                print("Filename:",file)
            counter += 1
            progress = counter / max_images_to_search
            if counter % 200 == 0:
                st.write(f"Processed {counter} images")
            
            progress_bar.progress(progress)
            fingerprint_image = cv2.imread("SOCOFing/Real/" + file)
            sift = cv2.SIFT_create()

            keypoints_1 , descriptors_1 = sift.detectAndCompute(sample,None)
            keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

            matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10},
                                            {}).knnMatch(descriptors_1, descriptors_2, k=2)

            match_points = []

            for p, q in matches:
                if p.distance < 0.1 * q.distance:
                    match_points.append(p)

            keypoints = 0
            if len(keypoints_1) < len(keypoints_2):
                keypoints = len(keypoints_1)
            else:
                keypoints = len(keypoints_2)

            if len(match_points) / keypoints * 100 > best_score:
                best_score = len(match_points) / keypoints * 100
                filename = file
                image = fingerprint_image
                kp1 , kp2 , mp = keypoints_1, keypoints_2 , match_points

        print("\n*********************************\nORIGINAL FILENAME: ",input_filename,"\nBEST MATCH : " + filename)
        print("SCORE : " + str(best_score))
        print("\n*********************************")
        st.write("\n*********************************\nORIGINAL FILENAME: ", input_filename)
        st.write("\nBEST MATCH : " + filename)
        st.write("SCORE : " + str(best_score))


        result = cv2.drawMatches(sample,kp1,image,kp2,mp,None)
        result = cv2.resize(result,(1530,850),fx=4,fy=4) # For Fullscreen (1530,850)
        st.write("\n*********************************")
        st.write("\n Result")
        st.image(result, caption="Matched Image", use_column_width=True)
        st.write("\n*********************************")
        #cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        st.warning("Please upload an image to start the matching process.")

if __name__ == "__main__":
    main()