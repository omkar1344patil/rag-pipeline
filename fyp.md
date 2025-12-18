# Final-Year-Project

Final Year Ideas

- AI glasses for the visually impaired to assist with their day-to-day tasks. Computer Vision glasses which can identify things for the visually impaired and give them a intructions in day-today activities. It should help with navigation, street signs, text and document reading, social identification with facial recognition, daily assistance like currency management, finding specific things. The backend would work with a raspberry pi and a camera module with local img recog and systems within the raspberry pi. this system would be attached to a glass frame to try it out. 80-90% work could be done as a normal app as well. 20-30% is the hardware part if i decide to implement it.
- Why is it helpful? it’s solving a major problem almost 200M+ People have. and this solution would be cost effective as compared to current industry standards of vision glasses. it’s genuinely life changing for the correct person.

Tasks

- title ready for next week, intro, motivation, expected final product until spring 2026
- targetting users, methodolgies, datasets to use / or personal, hardware costs, any modules to use. base plan of languages.
- find papers and relevancy

# Draft

Title : Vision AI Glasses, A Cost-efficient solution for Visually Impaired Individuals

Motivation :

Globally, 2.2 billion people have vision impairment, with at least 1 billion having preventable or unaddressed vision loss. 

( ﻿https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment , https://www.who.int/publications-detail-redirect/world-report-on-vision)

This creates an enormous economic burden with annual productivity losses of US$ 411 billion. ﻿

(https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment)﻿ Despite this massive need, nearly 90% of visually impaired people requiring assistive technologies cannot access them ﻿ ( https://www.zionmarketresearch.com/report/assistive-technologies-for-visually-impaired-market﻿), primarily due to cost barriers. Existing smart glasses like OrCam MyEye ($4,250) and eSight ($5,000-$6,000) remain prohibitively expensive https://lighthouseguild.org/news/fantastic-new-tech-for-people-with-low-vision-or-blindness/ ﻿, while 89% of visually impaired people live in low- and middle-income countries where such devices are completely inaccessible. 

Impact :

By leveraging Raspberry Pi and local computer vision processing, this project aims to deliver comprehensive assistive functionality, navigation, text reading, object identification, facial recognition, at a price point 10-20x lower than commercial alternatives. Local edge processing enables fast, real-time responses without internet dependency ﻿Raspberry Pi﻿, making it viable for users in underserved areas. Cost-effective assistive technologies directly increase independence for individuals with visual impairments in all areas of life , and research confirms that eye health interventions deliver exceptional value with a $28 return per dollar invested.  This solution doesn’t just provide technology, it restores independence, enables employment opportunities, and transforms quality of life for millions who currently have no access to such life-changing assistance.

Optimization: INT8 quantization, model pruning for real-time performance

Architecture: Multi-threaded (separate camera, vision, audio threads), modular design

## Final Deliverable Product :

AI-powered Glasses based on a Raspberry Pi 4, along with modules like camera, microphone, small speakers, all powered by a battery pack (powerbank). The Main system (Raspberry Pi) will be pocket mounted connected via cables to the glasses.

Features :

Voice assistance abilities to call the module and , 

“Hey Vision, read this text for me”

“Hey Vision, Remember this person as Omkar” (Saves the person’s face in local db to recognise later)

## Ideal Customers:

- Vision Impaired people especially in less-economical regions of the world like South and South East Asia and Africa where household income is <$100 per month.
- Students age from 3 onwards, Aged people over 50, People who can’t afford latest high-tech assistive technology solutions

# Methodology :

## System Design :

Hardware → Raspberry Pi 4, camera modules, audio systems based on glasses

Software architecture -> Multi-threaded Python script (separate threads: camera capture, vision processing, audio I/O)

Model Development :

- Object Detection: YOLOv5n (COCO dataset, 15-25 FPS)
- OCR: Tesseract/EasyOCR
- Face Recognition: face_recognition library (dlib-based)
- Currency: Custom CNN on Britain’s currency dataset.

Voice Features:

- Wake word capabilities using Porcupine wake module
- Speech-to-text: Vosk
- Text-to-speech: Piper TTS

Libraries: PyTorch/TFLite, OpenCV, numpy, pyaudio

Hardware Integration :

Camera module and Microphone attached to the glasses, connected to the raspberry pi via a cable. The Raspberry PI will be pocket mounted along with it’s own case.

Hardware :

| Raspberry Pi 5 | Provided by University |  |
| --- | --- | --- |
| Camera Module |  |  |
|  |  |  |

Raspberry Pi 4 - 4GB - £48 ([Link](https://cpc.farnell.com/raspberry-pi/rpi4-modbp-4gb/raspberry-pi-4-model-b-4gb/dp/SC15185?mckv=sshopping_dc|pcrid|605262956803|kword||match||plid||slid||product|SC15185|pgrid|138313687415|ptaid|pla-2241718292126|&CMP=KNC-GUK-CPC-SHOPPING-9262013734-138313687415-SC15185&s_kwcid=AL!5616!3!605262956803!!!network}!2241718292126!&gad_source=4&gad_campaignid=9262013734&gbraid=0AAAAAD_m6B3Ru8dB2x8D1IUiT6FWe7oMc&gclid=CjwKCAjwup3HBhAAEiwA7euZuqYRPhifWQXSW_2lojrOqx220m_wFlHcwru2dba4WtdZ1XYTPagr5xoC-GUQAvD_BwE)) (https://uk.webuy.com/product-detail?id=SDESRPFPIB04B&categoryName=SINGLE-BOARD-COMPUTERS&superCatName=COMPUTING&title=)

 - £10 - (https://uk.webuy.com/product-detail?id=0640522710881)

Microphone - £4 - https://thepihut.com/products/wired-miniature-electret-microphone?variant=27739691473&country=GB&currency=GBP&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&gad_source=1&gad_campaignid=11673057096&gbraid=0AAAAADfQ4GFPLDVK7xP37F5ss4WbUwFvq&gclid=CjwKCAjwup3HBhAAEiwA7euZuqdL28gzN0GK0LZIuyyIVmPDnpfzZJJ3SmF0EKuB3nljXzNvtd0L1RoC2dYQAvD_BwE

Powerbank - I already have one spare

Glasses Frame - I have one

Casing for Raspberry Pi - £4 - https://www.amazon.co.uk/components-Raspberry-Protective-Model-Transparent/dp/B094W2WTJW?source=ps-sl-shoppingads-lpcontext&ref_=fplfs&psc=1&smid=A2717MKXZVZ1ZW

Total Cost → £66 + (Miscellaneous costs of £5-£10)

- Feedback from Jia:
    
    Priority of messages delivered
    time of msg delivered
    how its delivered
    
    write scenarios
    
    To tackle Priorities while delivering messages, it’s better to introduce modes. like outdoor mode, university mode, indoor mode, 
    
- Chat with Yasmin (13/10)
    - Discussed the Ai glasses idea with her, elaborated on how the idea is going to be developed with software and hardware considerations. She commented positively “sounds very good”
    - Another thing to focus from this chat was considering the method of evaluation of the project i.e Finding the person (visually impaired). i told her that my supervisor advised to use scenarios. Yasmin mentioned getting the Ethical Approval would be tricky. although she suggested to address the evaluation by using stereotypes and personas from HCI.
- Feedback from Jia 23/10
    - Combine points 2.1.1 - 2.1.5
    - dataset collection before in initial planning
    - make sure to tell exact ml model i.e. make each objective descriptive 0bj4 (too many sub objective) and make more informative and we know the models, ui , data
    - rewrite estimates for entire objective
    - merge subobjectives
    - combine hardware testing while building the models
    - move the testing part to the testing sections
    - implement inline citations and mention url for companies - footnotes
    - mention recent developments, existing technologies and still missing or needs development (research papers)
    - mention data, models in methodology
    - remove money parts from methodology
    - just mention cost effective
    - gant chart for project plan
    - in evaluation, matrix testing part and remove 4 entirely
    - mention future work testing with visually impaired so not ethical approval required

### Raspberry Pi 3 - from university IT labs

Square Speakers : https://www.aliexpress.com/item/1005009962950710.html?src=google&pdp_npi=4%40dis!GBP!3.43!1.70!!!!!%40!12000050701509034!ppc!!!&snps=y&snpsid=1&src=google&albch=shopping&acnt=742-864-1166&isdl=y&slnk=&plac=&mtctp=&albbt=Google_7_shopping&aff_platform=google&aff_short_key=_oDeeeiG&gclsrc=aw.ds&&albagn=888888&&ds_e_adid=&ds_e_matchtype=&ds_e_device=c&ds_e_network=x&ds_e_product_group_id=&ds_e_product_id=en1005009962950710&ds_e_product_merchant_id=5551326180&ds_e_product_country=GB&ds_e_product_language=en&ds_e_product_channel=online&ds_e_product_store_id=&ds_url_v=2&albcp=22435797343&albag=&isSmbAutoCall=false&needSmbHouyi=false&gad_source=1&gad_campaignid=22432265180&gbraid=0AAAAA99aYpc0g20liSMF_oSfygbQAxM1z&gclid=Cj0KCQiAiKzIBhCOARIsAKpKLANZUmZrWevgIiCg94AdQGBfxdb9Hs_LQ49Ix8_mRyUGqdNV-nj5s9kaAn57EALw_wcB

Glasses Thick Frame: https://www.aliexpress.com/item/1005007733569540.html?src=google&pdp_npi=4%40dis!GBP!6.43!2.72!!!!!%40!12000042023834553!ppc!!!&snps=y&snpsid=1&src=google&albch=shopping&acnt=742-864-1166&isdl=y&slnk=&plac=&mtctp=&albbt=Google_7_shopping&aff_platform=google&aff_short_key=_oDeeeiG&gclsrc=aw.ds&&albagn=888888&&ds_e_adid=&ds_e_matchtype=&ds_e_device=c&ds_e_network=x&ds_e_product_group_id=&ds_e_product_id=en1005007733569540&ds_e_product_merchant_id=5551326180&ds_e_product_country=GB&ds_e_product_language=en&ds_e_product_channel=online&ds_e_product_store_id=&ds_url_v=2&albcp=22435797343&albag=&isSmbAutoCall=false&needSmbHouyi=false&gad_source=1&gad_campaignid=22432265180&gbraid=0AAAAA99aYpc0g20liSMF_oSfygbQAxM1z&gclid=Cj0KCQiAiKzIBhCOARIsAKpKLAPW2t_e-jrWqMt4J_9x08hPVp_7NdOU-eOZGO1mu9wtUrcqbv4TAxYaAucQEALw_wcB

Camera Module : https://thepihut.com/products/raspberry-pi-camera-module-3?variant=42305752039619&country=GB&currency=GBP&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&srsltid=AfmBOooCLHSgT7-4WwoRyTHPh3lbMwP_mbXI7Kth0kiv5MZUARAPLqr59yk

1m Camera Cable : https://thepihut.com/products/flex-cable-for-raspberry-pi-camera-or-display-1-meter?variant=13930092036&country=GB&currency=GBP&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&srsltid=AfmBOoqTF0MRnoYtT0UmNL8tmMTLcJWXQYYvTZdKRRrHQy5mN5q_6nNizF8

Camera Cable extender : https://thepihut.com/products/camera-cable-joiner-extender-for-raspberry-pi

3.5mm jack to bare wire : https://www.aliexpress.com/item/1005005892049362.html?spm=a2g0o.tesla.0.0.589dWsTjWsTjJQ&pdp_npi=6%40dis%21GBP%21%EF%BF%A11.72%21%EF%BF%A10.78%21%21%21%21%21%40210384b917623735314253504e2435%2112000034731682285%21btfpre%21%21%21%211%210%21&afTraceInfo=1005005892049362__pc__c_ppc_item_bridge_pc_same_wf__AHxWgoG__1762373531757

tortoise tts for text to speech generation

![image.png](Final-Year-Project/image.png)

Items needed from Aliexpress:

- Earphones x 2 (to test directly after removing the plastic casing)
- Wire Casing 1m length - 20-25mm wide
- Thick Glasses frames
- Camera Converter 22pin to 15 pin
- Camera Cable 15 pin - 1 meter
- Camera Cable 22pin to 15pin

Personal -

- Magsafe case or Magnets for case
- Led Light for kitchen

Models

## Supporting Papers :

1. https://pmc.ncbi.nlm.nih.gov/articles/PMC12178407/
2. https://arxiv.org/abs/2405.07606
3. https://www.ijcrt.org/papers/IJCRT2504448.pdf
4. https://old.fruct.org/publications/acm25/files/Asf.pdf
5. https://iarjset.com/wp-content/uploads/2024/12/IARJSET.2024.111233.pdf

Project Proposal Feedback :

[file:///Users/omkar/Downloads/001257511_Omkar%20Patil%20-%20COMP1682%20Proposal%20FEEDBACK.docx](file:///Users/omkar/Downloads/001257511_Omkar%20Patil%20-%20COMP1682%20Proposal%20FEEDBACK.docx)

Why PIPERTSS is better for RPI : 

https://www.reddit.com/r/LocalLLaMA/comments/1f0awd6/best_local_open_source_texttospeech_and/

Feedback from Jia - 11th Dec

- Make the wake word and the reply line faster.
- You can customize the wake word if you have time to train a model
- Improve the Face recognition library - don’t use it out of the box
- Focus on the FYP Final demo - that’s the part to shine, “to show you’re smart”
    - Implement one or two scenarios showing how your product can fit in day-to-day works
    - Showcase how your product is realistic and solves problems for those users
    - You can enact a scenario showing how the product would support it users.
    - You can also showcase in another scenario on a critical problem which the product solves. A problem which can’t be solved without glasses. (May be like a truck approaching- etc think about it)
    - Keep your product ready - can’t be like i forgot the charger etc - (think of the power supply)
- Don’t use models out of the box - do some work of your own on it, contribute towards it
- Write more about research papers from the computer science domain.
    - For eg, mention how computer vision is used by others, mention how other researchers have worked around cv2 models, talk about why yours is better than the competitors
    - 

## Backup

- Finding Alternative trading signals using Satellite Imagery
- Real-time Fraud Detection in Credit (Banking Side of Project)