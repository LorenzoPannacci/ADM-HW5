from collections import defaultdict

def parse_input(input_filename):
    with open(input_filename) as f:
        lines = f.readlines()

    # First line is n, m and s values respectively
    n, m, s = [int(x) for x in lines[0].split()]

    # Second line is set of skills needed for competition
    list_of_skills = lines[1].split()

    athlete_dict = defaultdict(dict)
    i = 2

    # Rest of the lines is athlete's and their skills with scores
    while i < len(lines):
        # If only id is present, that is the new athlete
        if len(lines[i].split()) == 1:
            athlete_id = int(lines[i].strip())
        # This is the skill and corresponding score for the athlete_id
        else:
            skill, score = lines[i].split()
            athlete_dict[athlete_id][skill] = int(score)
        i += 1
    return n, m, s, list_of_skills, athlete_dict

def dfs(athlete_dict, list_of_skills, used_athletes, current_index, current_score):
    # We reached the end of skills needed
    if current_index == len(list_of_skills):
        return current_score

    max_score = 0

    for athlete, skills in athlete_dict.items():
        # Each athlete must be unique, so only those that are not in used_athletes are interesting for consideration
        if athlete not in used_athletes:
            # Add that athlete to the list
            used_athletes.add(athlete)
            # Because we will consider each athlete for each skill, we will take that athlete into consideration
            # If he does not have that skill, in that case, give the score 0
            skill_score = skills.get(list_of_skills[current_index], 0)
            # Recursive call with increased current_index and current_score
            score = dfs(
                athlete_dict,
                list_of_skills,
                used_athletes,
                current_index + 1,
                current_score + skill_score,
            )
            # Get the score that is maximum
            max_score = max(max_score, score)
            # Now remove the athlete from used_athletes, so we can check the new skill for this athlete
            used_athletes.remove(athlete)

    return max_score


# Wrapper that takes input the file where the problem is stored using the DFS approach
def give_maximum_score_dfs(input_filename):
    n, m, s, list_of_skills, athlete_dict = parse_input(input_filename)
    used_athletes = set()
    return dfs(athlete_dict, list_of_skills, used_athletes, 0, 0)

def give_maximum_score_greedy(input_filename):
    n, m, s, list_of_skills, athlete_dict = parse_input(input_filename)
    skill_dict = defaultdict(list)

    # Creation of skills dictionary
    for athlete_id, skills in athlete_dict.items():
        for skill, score in skills.items():
            skill_dict[skill].append((score, athlete_id))

    # Making the athletes with highest scores be on top
    for skill in skill_dict:
        skill_dict[skill].sort(reverse=True)

    assigned_athletes = set()
    total_score = 0

    # Now we go through all the skills and take the athletes with the highest score
    for skill in list_of_skills:
        for score, athlete_id in skill_dict[skill]:
            # If they are assigned already, don't take them into consideration
            if athlete_id not in assigned_athletes:
                total_score += score
                assigned_athletes.add(athlete_id)
                break

    return total_score