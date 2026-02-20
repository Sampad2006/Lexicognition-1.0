This is a fixed and better version of lexicognition 0.1. which was meant as a group hackathon project.
This imporved version is completely a personal project.
I know the names arent consistent
Plans :
Try stopping sessions in between or adding new pdfs to change questions at one moment


Last weeks trials and efforts:
Tried to migrate from this monolith to an api key pool(3 apis) managed by redis which checks the ratelimits and usage of each api key.
Works on localhost.
changes yet to be made on community cloud

Also instead of a single llmcreation broke it into steps inside both question_generator and answer_grader to create new instances every time, which utilzes the new load balancing pool work too.

Community Cloud issues: added session_id in constructor, so each user gets own unique folder instead of previous shared chroma_db folder

few hashing issues when hashing a list which wasnt allowed by default in python, so upgraded to hashlib-deterministic hashing
