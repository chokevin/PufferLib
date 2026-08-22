#pragma once

static inline int clampi(int value, int low, int high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static inline float clampf(float value, float low, float high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static inline int max_health(const State* state) {
    return 8 + state->player_strength;
}

static inline int max_food(const State* state) {
    return 7 + 2 * state->player_dexterity;
}

static inline int max_drink(const State* state) {
    return 7 + 2 * state->player_dexterity;
}

static inline int max_energy(const State* state) {
    return 7 + 2 * state->player_dexterity;
}

static inline int max_mana(const State* state) {
    return 6 + 3 * state->player_intelligence;
}

static inline bool fighting_boss(const State* state) {
    return state->player_level == NUM_LEVELS - 1;
}

static inline int jax_index(int index, int size) {
    if (index < 0) {
        index += size;
    }
    if (index < 0) {
        return 0;
    }
    if (index >= size) {
        return size - 1;
    }
    return index;
}

static inline bool boss_vulnerable(const State* state) {
    if (state->boss_timestep_to_spawn_this_round > 0) {
        return false;
    }
    int level = jax_index(state->player_level, NUM_LEVELS);
    for (int i = 0; i < MAX_MELEE_MOBS; i++) {
        if (state->melee_mobs[level].mask[i]) {
            return false;
        }
    }
    for (int i = 0; i < MAX_RANGED_MOBS; i++) {
        if (state->ranged_mobs[level].mask[i]) {
            return false;
        }
    }
    return true;
}

static inline bool has_beaten_boss(const State* state) {
    return state->boss_progress >= NUM_LEVELS - 1;
}

static inline void action_to_direction(int action, int direction[2]) {
    direction[0] = 0;
    direction[1] = 0;

    if (action == ACTION_LEFT) {
        direction[1] = -1;
    } else if (action == ACTION_RIGHT) {
        direction[1] = 1;
    } else if (action == ACTION_UP) {
        direction[0] = -1;
    } else if (action == ACTION_DOWN) {
        direction[0] = 1;
    }
}

static inline bool is_solid_block(int block) {
    switch (block) {
    case BLOCK_STONE:
    case BLOCK_TREE:
    case BLOCK_COAL:
    case BLOCK_IRON:
    case BLOCK_DIAMOND:
    case BLOCK_CRAFTING_TABLE:
    case BLOCK_FURNACE:
    case BLOCK_PLANT:
    case BLOCK_RIPE_PLANT:
    case BLOCK_WALL:
    case BLOCK_WALL_MOSS:
    case BLOCK_STALAGMITE:
    case BLOCK_RUBY:
    case BLOCK_SAPPHIRE:
    case BLOCK_CHEST:
    case BLOCK_FOUNTAIN:
    case BLOCK_FIRE_TREE:
    case BLOCK_ENCHANTMENT_TABLE_FIRE:
    case BLOCK_ENCHANTMENT_TABLE_ICE:
    case BLOCK_GRAVE:
    case BLOCK_GRAVE2:
    case BLOCK_GRAVE3:
    case BLOCK_NECROMANCER:
        return true;
    default:
        return false;
    }
}

static inline bool mob_at(const State* state, int level, int row, int col) {
    if ((unsigned)row >= MAP_SIZE || (unsigned)col >= MAP_SIZE) {
        return false;
    }
    return (state->mob_bits[level][row] >> col) & 1ull;
}

static inline void set_mob_bit(State* state, int level, int row, int col, bool on) {
    if ((unsigned)row >= MAP_SIZE || (unsigned)col >= MAP_SIZE) {
        return;
    }
    uint64_t bit = 1ull << col;
    if (on) {
        state->mob_bits[level][row] |= bit;
    } else {
        state->mob_bits[level][row] &= ~bit;
    }
}

static inline void move_mob_occupancy(
    State* state, int level, int old_row, int old_col, int new_row, int new_col, bool keep
) {
    set_mob_bit(state, level, old_row, old_col, false);
    if (keep) {
        set_mob_bit(state, level, new_row, new_col, true);
    }
}

static inline bool mobs_at(const Mobs* mobs, int slots, int row, int col, int* slot) {
    for (int i = 0; i < slots; i++) {
        if (mobs->mask[i]
            && mobs->position[i][0] == row
            && mobs->position[i][1] == col) {
            *slot = i;
            return true;
        }
    }
    return false;
}

static inline bool valid_player_position(const State* state, int row, int col) {
    if (row < 0 || row >= MAP_SIZE || col < 0 || col >= MAP_SIZE) {
        return false;
    }

    int level = clampi(state->player_level, 0, NUM_LEVELS - 1);
    int block = state->map[level][row][col];
    if (is_solid_block(block)) {
        return false;
    }
    if (block == BLOCK_WATER || block == BLOCK_LAVA) {
        return false;
    }
    return !mob_at(state, level, row, col);
}

static inline Mobs* mobs_for_class(State* state, int level, int mob_class) {
    if (mob_class == MOB_PASSIVE) {
        return &state->passive_mobs[level];
    }
    if (mob_class == MOB_RANGED) {
        return &state->ranged_mobs[level];
    }
    return &state->melee_mobs[level];
}

static inline bool find_mob_at(
    const State* state,
    int level,
    int row,
    int col,
    int* mob_class,
    int* slot
) {
    if (mobs_at(&state->melee_mobs[level], MAX_MELEE_MOBS, row, col, slot)) {
        *mob_class = MOB_MELEE;
        return true;
    }
    if (mobs_at(&state->passive_mobs[level], MAX_PASSIVE_MOBS, row, col, slot)) {
        *mob_class = MOB_PASSIVE;
        return true;
    }
    if (mobs_at(&state->ranged_mobs[level], MAX_RANGED_MOBS, row, col, slot)) {
        *mob_class = MOB_RANGED;
        return true;
    }
    return false;
}

static inline bool mob_can_move_on(int mob_class, int type_id, int block) {
    static const bool blocked[NUM_MOB_TYPES][3][3] = {
        {{0,1,1},{0,1,1},{0,1,1}}, {{0,0,0},{0,1,1},{0,1,1}},
        {{0,1,1},{0,1,1},{0,1,1}}, {{0,1,1},{0,0,1},{0,1,1}},
        {{0,1,1},{0,1,1},{0,1,1}}, {{0,1,1},{0,1,1},{1,0,1}},
        {{0,1,1},{0,1,1},{0,0,0}}, {{0,1,1},{0,1,1},{0,0,0}},
    };
    if (is_solid_block(block)) return false;
    int terrain = block == BLOCK_WATER ? 1 : (block == BLOCK_LAVA ? 2 : 0);
    return !blocked[clampi(type_id,0,7)][clampi(mob_class,0,2)][terrain];
}

static inline bool valid_typed_mob_position(
    const State* state,
    int level,
    int mob_class,
    int type_id,
    int row,
    int col,
    int old_row,
    int old_col
) {
    if (row < 0 || row >= MAP_SIZE || col < 0 || col >= MAP_SIZE) {
        return false;
    }
    if (row == state->player_position[0] && col == state->player_position[1]) {
        return false;
    }
    if (!mob_can_move_on(mob_class, type_id, state->map[level][row][col])) {
        return false;
    }
    return !mob_at(state, level, row, col) || (row == old_row && col == old_col);
}

static inline float mob_base_health(int mob_class, int type_id) {
    static const float passive_health[NUM_MOB_TYPES] = {3, 4, 6, 8, 0, 0, 0, 0};
    static const float melee_health[NUM_MOB_TYPES] = {5, 7, 9, 11, 12, 20, 20, 24};
    static const float ranged_health[NUM_MOB_TYPES] = {3, 5, 6, 8, 12, 4, 14, 16};
    int idx = clampi(type_id, 0, NUM_MOB_TYPES - 1);
    if (mob_class == MOB_PASSIVE) return passive_health[idx];
    if (mob_class == MOB_RANGED) return ranged_health[idx];
    return melee_health[idx];
}

typedef struct { float physical, fire, ice; } Damage;

static inline Damage player_attack_damage_vector(const State* state) {
    static const float base_damage[5] = {1, 2, 3, 5, 8};
    float base = base_damage[clampi(state->inventory.sword, 0, 4)];
    float physical = base * (1.0f + 0.25f * (float)(state->player_strength - 1));
    float magic = base * 0.5f * (1.0f + 0.05f * (float)(state->player_intelligence - 1));
    return (Damage){physical, state->sword_enchantment == 1 ? magic : 0, state->sword_enchantment == 2 ? magic : 0};
}

static inline Damage mob_damage_vector(int type, int mob_class) {
    static const float damage[NUM_MOB_TYPES][4][3] = {
        {{0,0,0},{2,0,0},{0,0,0},{2,0,0}}, {{0,0,0},{4,0,0},{0,0,0},{4,0,0}},
        {{0,0,0},{3,0,0},{0,0,0},{0,3,0}}, {{0,0,0},{5,0,0},{0,0,0},{0,0,3}},
        {{0,0,0},{6,0,0},{0,0,0},{5,0,0}}, {{0,0,0},{6,1,1},{0,0,0},{4,3,3}},
        {{0,0,0},{3,5,0},{0,0,0},{3,5,0}}, {{0,0,0},{4,0,5},{0,0,0},{4,0,5}},
    };
    const float* d = damage[clampi(type, 0, 7)][clampi(mob_class, 0, 3)];
    return (Damage){d[0], d[1], d[2]};
}

static inline float damage_to_mob(Damage damage, int type, int mob_class) {
    static const float defense[NUM_MOB_TYPES][4][3] = {
        {{0,0,0},{0,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{0,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{0,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{0,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{.5f,0,0},{.5f,0,0},{0,0,0}},
        {{0,0,0},{.2f,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{.9f,1,0},{.9f,1,0},{0,0,0}},
        {{0,0,0},{.9f,0,1},{.9f,0,1},{0,0,0}},
    };
    const float* d = defense[clampi(type, 0, 7)][clampi(mob_class, 0, 3)];
    return damage.physical * (1-d[0]) + damage.fire * (1-d[1]) + damage.ice * (1-d[2]);
}

static inline float damage_to_player(const State* state, Damage damage) {
    float physical_defense = 0, fire_defense = 0, ice_defense = 0;
    for (int i = 0; i < 4; i++) {
        physical_defense += 0.1f * state->inventory.armour[i];
        fire_defense += 0.2f * (state->armour_enchantments[i] == 1);
        ice_defense += 0.2f * (state->armour_enchantments[i] == 2);
    }
    float coeff = fighting_boss(state) ? 1.5f : 1.0f;
    return coeff * (damage.physical * (1-physical_defense) + damage.fire * (1-fire_defense) + damage.ice * (1-ice_defense));
}

static inline int defeat_achievement(int mob_class, int type_id, int level) {
    (void)level;
    static const int achievements[3][8] = {
        {ACH_EAT_COW, ACH_EAT_BAT, ACH_EAT_SNAIL, 0,0,0,0,0},
        {ACH_DEFEAT_ZOMBIE, ACH_DEFEAT_GNOME_WARRIOR, ACH_DEFEAT_ORC_SOLIDER, ACH_DEFEAT_LIZARD, ACH_DEFEAT_KNIGHT, ACH_DEFEAT_TROLL, ACH_DEFEAT_PIGMAN, ACH_DEFEAT_FROST_TROLL},
        {ACH_DEFEAT_SKELETON, ACH_DEFEAT_GNOME_ARCHER, ACH_DEFEAT_ORC_MAGE, ACH_DEFEAT_KOBOLD, ACH_DEFEAT_ARCHER, ACH_DEFEAT_DEEP_THING, ACH_DEFEAT_FIRE_ELEMENTAL, ACH_DEFEAT_ICE_ELEMENTAL},
    };
    return achievements[clampi(mob_class, 0, 2)][clampi(type_id, 0, 7)];
}

static inline bool damage_mob_at(
    State* state, int level, int row, int col, float damage,
    bool can_eat, bool can_get_achievement
) {
    int mob_class;
    int slot;
    if (!find_mob_at(state, level, row, col, &mob_class, &slot)) {
        return false;
    }
    Mobs* mobs = mobs_for_class(state, level, mob_class);
    if (!mobs->mask[slot]) {
        return false;
    }

    mobs->health[slot] -= damage;
    if (mobs->health[slot] > 0.0f) {
        return true;
    }

    int type_id = mobs->type_id[slot];
    mobs->mask[slot] = false;
    set_mob_bit(state, level, row, col, false);
    state->monsters_killed[level] += mob_class == MOB_PASSIVE ? 0 : 1;
    if (can_get_achievement) {
        state->achievements[defeat_achievement(mob_class, type_id, level)] = 1;
    }

    if (mob_class == MOB_PASSIVE && can_eat) {
        state->player_food = clampi(state->player_food + 6, 0, max_food(state));
        state->player_hunger = 0.0f;
    }
    return true;
}

static inline bool attack_mob_at(State* state, int level, int row, int col, bool can_eat) {
    int mob_class;
    int slot;
    if (!find_mob_at(state, level, row, col, &mob_class, &slot)) {
        return false;
    }
    Mobs* mobs = mobs_for_class(state, level, mob_class);
    Damage vector = player_attack_damage_vector(state);
    return damage_mob_at(
        state, level, row, col,
        damage_to_mob(vector, mobs->type_id[slot], mob_class),
        can_eat, true
    );
}

static inline float projectile_damage(int projectile_type, bool from_player) {
    (void)from_player;
    Damage d = mob_damage_vector(projectile_type, MOB_PROJECTILE);
    return d.physical + d.fire + d.ice;
}

static inline Damage player_projectile_damage(const State* state, int type) {
    Damage damage = mob_damage_vector(type, MOB_PROJECTILE);
    bool arrow = type == PROJECTILE_ARROW || type == PROJECTILE_ARROW2;
    if (arrow && state->bow_enchantment == 1) damage.fire += damage.physical * 0.5f;
    if (arrow && state->bow_enchantment == 2) damage.ice += damage.physical * 0.5f;
    float coeff = arrow ? 1.0f + 0.2f * (state->player_dexterity - 1) :
        ((type == PROJECTILE_FIREBALL || type == PROJECTILE_ICEBALL) ? 1.0f + 0.5f * (state->player_intelligence - 1) : 1.0f);
    damage.physical *= coeff; damage.fire *= coeff; damage.ice *= coeff;
    return damage;
}

static inline bool spawn_projectile(
    State* state,
    bool from_player,
    int projectile_type,
    int row,
    int col,
    int dir_row,
    int dir_col
) {
    int level = clampi(state->player_level, 0, NUM_LEVELS - 1);
    Mobs* projectiles = from_player ? &state->player_projectiles[level] : &state->mob_projectiles[level];
    int (*directions)[MAX_PLAYER_PROJECTILES][2] =
        from_player ? state->player_projectile_directions : state->mob_projectile_directions;
    for (int i = 0; i < MAX_PLAYER_PROJECTILES; i++) {
        if (projectiles->mask[i]) {
            continue;
        }
        projectiles->position[i][0] = row;
        projectiles->position[i][1] = col;
        projectiles->health[i] = projectile_damage(projectile_type, from_player);
        projectiles->attack_cooldown[i] = 0;
        projectiles->type_id[i] = projectile_type;
        projectiles->mask[i] = true;
        directions[level][i][0] = dir_row;
        directions[level][i][1] = dir_col;
        return true;
    }
    return false;
}

static inline int read_map_block(const State* state, int level, int row, int col) {
    return state->map[jax_index(level, NUM_LEVELS)]
        [jax_index(row, MAP_SIZE)]
        [jax_index(col, MAP_SIZE)];
}

static inline bool projectile_in_mob(const State* state, int level, int row, int col) {
    int map_level = jax_index(level, NUM_LEVELS);
    int map_row = jax_index(row, MAP_SIZE);
    int map_col = jax_index(col, MAP_SIZE);
    bool player_here = state->player_position[0] == row
        && state->player_position[1] == col;
    return mob_at(state, map_level, map_row, map_col) || player_here;
}

static inline void scatter_set_block(State* state, int level, int row, int col, int block) {
    int mapped_level;
    int mapped_row;
    int mapped_col;
    if (!scatter_index(level, NUM_LEVELS, &mapped_level)
        || !scatter_index(row, MAP_SIZE, &mapped_row)
        || !scatter_index(col, MAP_SIZE, &mapped_col)) {
        return;
    }
    set_block(state, mapped_level, mapped_row, mapped_col, block);
}

static inline void move_mob_projectile_slot(State* state, int level, int slot) {
    Mobs* projectiles = &state->mob_projectiles[level];
    bool alive = projectiles->mask[slot];
    if (!alive) {
        return;
    }
    int old_row = projectiles->position[slot][0];
    int old_col = projectiles->position[slot][1];
    int proposed_row = old_row + state->mob_projectile_directions[level][slot][0];
    int proposed_col = old_col + state->mob_projectile_directions[level][slot][1];
    bool proposed_in_player = proposed_row == state->player_position[0]
        && proposed_col == state->player_position[1];
    bool proposed_in_bounds = proposed_row >= 0 && proposed_row < MAP_SIZE
        && proposed_col >= 0 && proposed_col < MAP_SIZE;
    int proposed_block = read_map_block(state, level, proposed_row, proposed_col);
    bool in_wall = is_solid_block(proposed_block) && proposed_block != BLOCK_WATER;
    bool in_mob = projectile_in_mob(state, level, proposed_row, proposed_col);
    bool keep_moving = proposed_in_bounds && !in_wall && !in_mob;
    bool hit_player = (
        (old_row == state->player_position[0] && old_col == state->player_position[1])
        || proposed_in_player
    ) && alive;
    keep_moving = keep_moving && !hit_player;
    bool keep = keep_moving && alive;
    bool hit_bench = proposed_block == BLOCK_FURNACE
        || proposed_block == BLOCK_CRAFTING_TABLE;
    int new_block = (hit_bench && alive) ? BLOCK_PATH : proposed_block;

    projectiles->position[slot][0] = proposed_row;
    projectiles->position[slot][1] = proposed_col;
    projectiles->mask[slot] = keep;
    if (hit_player) {
        state->player_health -= damage_to_player(
            state, mob_damage_vector(projectiles->type_id[slot], MOB_PROJECTILE));
        state->is_sleeping = false;
        state->is_resting = false;
    }
    scatter_set_block(state, level, proposed_row, proposed_col, new_block);
}

static inline void move_player_projectile_slot(State* state, int level, int slot) {
    Mobs* projectiles = &state->player_projectiles[level];
    bool alive = projectiles->mask[slot];
    if (!alive) {
        return;
    }
    int old_row = projectiles->position[slot][0];
    int old_col = projectiles->position[slot][1];
    int proposed_row = old_row + state->player_projectile_directions[level][slot][0];
    int proposed_col = old_col + state->player_projectile_directions[level][slot][1];
    Damage vector = player_projectile_damage(state, projectiles->type_id[slot]);
    if (!alive) {
        vector.physical = 0.0f;
        vector.fire = 0.0f;
        vector.ice = 0.0f;
    }

    bool hit_old = false;
    int mob_class;
    int mob_slot;
    if (find_mob_at(state, level, old_row, old_col, &mob_class, &mob_slot)) {
        Mobs* target = mobs_for_class(state, level, mob_class);
        hit_old = damage_mob_at(
            state, level, old_row, old_col,
            damage_to_mob(vector, target->type_id[mob_slot], mob_class),
            false, true
        );
    }

    Damage second = vector;
    if (hit_old) {
        second.physical = 0.0f;
        second.fire = 0.0f;
        second.ice = 0.0f;
    }
    bool hit_new = false;
    if (find_mob_at(state, level, proposed_row, proposed_col, &mob_class, &mob_slot)) {
        Mobs* target = mobs_for_class(state, level, mob_class);
        hit_new = damage_mob_at(
            state, level, proposed_row, proposed_col,
            damage_to_mob(second, target->type_id[mob_slot], mob_class),
            false, true
        );
    }

    bool proposed_in_bounds = proposed_row >= 0 && proposed_row < MAP_SIZE
        && proposed_col >= 0 && proposed_col < MAP_SIZE;
    int proposed_block = read_map_block(state, level, proposed_row, proposed_col);
    bool in_wall = is_solid_block(proposed_block) && proposed_block != BLOCK_WATER;
    bool keep = proposed_in_bounds && !in_wall && !hit_old && !hit_new && alive;
    projectiles->position[slot][0] = proposed_row;
    projectiles->position[slot][1] = proposed_col;
    projectiles->mask[slot] = keep;
}

static inline void update_projectile_set(State* state, bool from_player) {
    int level = jax_index(state->player_level, NUM_LEVELS);
    for (int i = 0; i < MAX_PLAYER_PROJECTILES; i++) {
        if (from_player) {
            move_player_projectile_slot(state, level, i);
        } else {
            move_mob_projectile_slot(state, level, i);
        }
    }
}

static inline int floor_mob_type(int level, int mob_class) {
    static const int types[NUM_LEVELS][3] = {
        {0, 0, 0}, {2, 2, 2}, {1, 1, 1}, {2, 3, 3}, {2, 4, 4},
        {1, 5, 5}, {1, 6, 6}, {1, 7, 7}, {0, 0, 0},
    };
    return types[clampi(level, 0, NUM_LEVELS - 1)][clampi(mob_class, 0, 2)];
}

static inline int pick_kth(int count, Rng key) {
    float draw = (float)count * (1.0f - rng_f32(key));
    float cumulative = 0.0f;
    for (int k = 0; k < count; k++) {
        cumulative += 1.0f;
        if (cumulative >= draw) {
            return k;
        }
    }
    return count - 1;
}

static inline int collect_spawn_cells(
    const State* state,
    int level,
    int min_exclusive,
    int max_exclusive,
    bool boss,
    bool water_only,
    int* rows,
    int* cols
) {
    const uint64_t* terrain = boss
        ? state->spawn_grave[level]
        : (water_only ? state->spawn_water[level] : state->spawn_land[level]);

    int pr = state->player_position[0];
    int pc = state->player_position[1];
    int limit = MOB_DESPAWN_DISTANCE - 1;
    int count = 0;
    for (int dr = -limit; dr <= limit; dr++) {
        int row = pr + dr;
        if ((unsigned)row >= MAP_SIZE) {
            continue;
        }
        uint64_t bits = terrain[row] & ~state->mob_bits[level][row];
        if (!bits) {
            continue;
        }
        for (int dc = -limit; dc <= limit; dc++) {
            int distance2 = dr * dr + dc * dc;
            if (distance2 <= min_exclusive || distance2 >= max_exclusive) {
                continue;
            }
            int col = pc + dc;
            if ((unsigned)col >= MAP_SIZE) {
                continue;
            }
            if (bits & (1ull << col)) {
                rows[count] = row;
                cols[count] = col;
                count++;
            }
        }
    }
    return count;
}

static inline bool pick_spawn_cell(
    const int* rows,
    const int* cols,
    int count,
    Rng key,
    int* out_row,
    int* out_col
) {
    if (count <= 0) {
        return false;
    }
    int chosen = pick_kth(count, key);
    *out_row = rows[chosen];
    *out_col = cols[chosen];
    return true;
}

static inline void spawn_into_slot(
    State* state,
    int level,
    Mobs* mobs,
    int slot,
    int mob_class,
    int type_id,
    int row,
    int col
) {
    mobs->position[slot][0] = row;
    mobs->position[slot][1] = col;
    mobs->health[slot] = mob_base_health(mob_class, type_id);
    mobs->mask[slot] = true;
    set_mob_bit(state, level, row, col, true);
}

static inline void count_and_empty(const Mobs* mobs, int slots, int* count, int* empty) {
    int n = 0;
    int first = 0;
    bool found = false;
    for (int i = 0; i < slots; i++) {
        n += mobs->mask[i] ? 1 : 0;
        if (!mobs->mask[i] && !found) {
            first = i;
            found = true;
        }
    }
    *count = n;
    *empty = first;
}

static inline void spawn_mobs(State* state, Rng rng) {
    int level = jax_index(state->player_level, NUM_LEVELS);
    bool boss = fighting_boss(state);
    int coeff = 1 + (state->monsters_killed[level] < MONSTERS_KILLED_TO_CLEAR_LEVEL ? 2 : 0);
    if (boss) {
        coeff *= (state->boss_timestep_to_spawn_this_round >= 1) ? 1000 : 0;
    }

    static const float chances[NUM_LEVELS][4] = {
        {0.1f, 0.02f, 0.05f, 0.1f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.0f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
    };

    int passive_count;
    int passive_slot;
    count_and_empty(&state->passive_mobs[level], MAX_PASSIVE_MOBS, &passive_count, &passive_slot);
    Rng passive_prob = rng_key(&rng);
    Rng passive_pos = rng_key(&rng);
    int passive_type = floor_mob_type(level, MOB_PASSIVE);
    state->passive_mobs[level].type_id[passive_slot] = passive_type;

    int melee_count;
    int melee_slot;
    count_and_empty(&state->melee_mobs[level], MAX_MELEE_MOBS, &melee_count, &melee_slot);
    int melee_type = floor_mob_type(boss ? state->boss_progress : level, MOB_MELEE);
    Rng melee_prob = rng_key(&rng);
    float night = 1.0f - state->light_level;
    float melee_chance = chances[level][1] + chances[level][3] * night * night;
    Rng melee_pos = rng_key(&rng);
    state->melee_mobs[level].type_id[melee_slot] = melee_type;

    int ranged_count;
    int ranged_slot;
    count_and_empty(&state->ranged_mobs[level], MAX_RANGED_MOBS, &ranged_count, &ranged_slot);
    int ranged_type = floor_mob_type(boss ? state->boss_progress : level, MOB_RANGED);
    Rng ranged_prob = rng_key(&rng);
    Rng ranged_pos = rng_key(&rng);
    state->ranged_mobs[level].type_id[ranged_slot] = ranged_type;

    bool try_passive = !boss && passive_count < MAX_PASSIVE_MOBS
        && rng_f32(passive_prob) < chances[level][0];
    bool try_melee = melee_count < MAX_MELEE_MOBS
        && rng_f32(melee_prob) < melee_chance * (float)coeff;
    bool try_ranged = ranged_count < MAX_RANGED_MOBS
        && rng_f32(ranged_prob) < chances[level][2] * (float)coeff;
    if (!try_passive && !try_melee && !try_ranged) {
        return;
    }

    int min_hostile = boss ? -1 : 81;
    int max_hostile = boss ? 37 : MOB_DESPAWN_DISTANCE * MOB_DESPAWN_DISTANCE;
    int spawn_rows[729];
    int spawn_cols[729];
    int row;
    int col;
    if (try_passive) {
        int n = collect_spawn_cells(
            state, level, 9, MOB_DESPAWN_DISTANCE * MOB_DESPAWN_DISTANCE,
            false, false, spawn_rows, spawn_cols);
        if (pick_spawn_cell(spawn_rows, spawn_cols, n, passive_pos, &row, &col)) {
            spawn_into_slot(
                state, level, &state->passive_mobs[level],
                passive_slot, MOB_PASSIVE, passive_type, row, col);
        }
    }
    if (try_melee) {
        int n = collect_spawn_cells(
            state, level, min_hostile, max_hostile,
            boss, false, spawn_rows, spawn_cols);
        if (pick_spawn_cell(spawn_rows, spawn_cols, n, melee_pos, &row, &col)) {
            spawn_into_slot(
                state, level, &state->melee_mobs[level],
                melee_slot, MOB_MELEE, melee_type, row, col);
        }
    }
    if (try_ranged) {
        int n = collect_spawn_cells(
            state, level, min_hostile, max_hostile,
            boss, ranged_type == 5, spawn_rows, spawn_cols);
        if (pick_spawn_cell(spawn_rows, spawn_cols, n, ranged_pos, &row, &col)) {
            spawn_into_slot(
                state, level, &state->ranged_mobs[level],
                ranged_slot, MOB_RANGED, ranged_type, row, col);
        }
    }
}

static inline void choose_direction(Rng key, int count, int direction[2]) {
    int choice = randint_at(key, 0u, 0, count);
    direction[0] = 0;
    direction[1] = 0;
    if (choice == 0) {
        direction[1] = -1;
    } else if (choice == 1) {
        direction[1] = 1;
    } else if (choice == 2) {
        direction[0] = -1;
    } else if (choice == 3) {
        direction[0] = 1;
    }
}

static inline int choose_player_axis(Rng key, int distance_row, int distance_col) {
    int total = distance_row + distance_col;
    if (total == 0) {
        return 1;
    }
    int maximum = distance_row > distance_col ? distance_row : distance_col;
    float weights[2] = {
        distance_row == maximum ? 1.0f / (float)total : 0.0f,
        distance_col == maximum ? 1.0f / (float)total : 0.0f,
    };
    float sum = weights[0] + weights[1];
    float draw = sum * (1.0f - rng_f32(key));
    return (weights[0] >= draw || sum == 0.0f) ? 0 : 1;
}

static inline int signi(int value) {
    if (value < 0) {
        return -1;
    }
    return value > 0 ? 1 : 0;
}

static inline void move_melee_slot(State* state, int level, int slot, Rng* rng) {
    Mobs* mobs = &state->melee_mobs[level];
    bool alive = mobs->mask[slot];
    if (!alive) {
        return;
    }
    int old_row = mobs->position[slot][0];
    int old_col = mobs->position[slot][1];
    int type_id = mobs->type_id[slot];
    int cooldown = mobs->attack_cooldown[slot];

    int random_dir[2];
    choose_direction(rng_key(rng), 4, random_dir);
    int distance_row = abs(state->player_position[0] - old_row);
    int distance_col = abs(state->player_position[1] - old_col);
    int axis = choose_player_axis(rng_key(rng), distance_row, distance_col);
    int player_dir[2] = {0, 0};
    if (axis == 0) {
        player_dir[0] = signi(state->player_position[0] - old_row);
    } else {
        player_dir[1] = signi(state->player_position[1] - old_col);
    }
    int dist = distance_row + distance_col;
    float chase_roll = rng_f32(rng_key(rng));
    bool chase = (dist < 10 || fighting_boss(state)) && chase_roll < 0.75f;
    int proposed_row = chase ? old_row + player_dir[0] : old_row + random_dir[0];
    int proposed_col = chase ? old_col + player_dir[1] : old_col + random_dir[1];
    bool attacking = dist == 1 && cooldown <= 0 && alive;
    if (attacking) {
        proposed_row = old_row;
        proposed_col = old_col;
        Damage damage = mob_damage_vector(type_id, MOB_MELEE);
        float sleep = 1.0f + 2.5f * (float)state->is_sleeping;
        damage.physical *= sleep;
        damage.fire *= sleep;
        damage.ice *= sleep;
        state->player_health -= damage_to_player(state, damage);
        state->achievements[ACH_WAKE_UP] = state->achievements[ACH_WAKE_UP] || state->is_sleeping;
        state->is_sleeping = false;
        state->is_resting = false;
    }
    int new_cooldown = attacking ? 5 : cooldown - 1;
    bool valid = valid_typed_mob_position(state, level, MOB_MELEE, type_id, proposed_row, proposed_col, old_row, old_col);
    int new_row = valid ? proposed_row : old_row;
    int new_col = valid ? proposed_col : old_col;
    bool keep = alive && (dist < MOB_DESPAWN_DISTANCE || fighting_boss(state));
    Rng unused;
    rng_split(*rng, &unused, rng);

    move_mob_occupancy(state, level, old_row, old_col, new_row, new_col, keep);
    mobs->position[slot][0] = new_row;
    mobs->position[slot][1] = new_col;
    mobs->attack_cooldown[slot] = new_cooldown;
    mobs->mask[slot] = keep;
}

static inline void move_passive_slot(State* state, int level, int slot, Rng* rng) {
    Mobs* mobs = &state->passive_mobs[level];
    bool alive = mobs->mask[slot];
    if (!alive) {
        return;
    }
    int old_row = mobs->position[slot][0];
    int old_col = mobs->position[slot][1];
    int type_id = mobs->type_id[slot];
    int direction[2];
    choose_direction(rng_key(rng), 8, direction);
    int proposed_row = old_row + direction[0];
    int proposed_col = old_col + direction[1];
    bool valid = valid_typed_mob_position(state, level, MOB_PASSIVE, type_id, proposed_row, proposed_col, old_row, old_col);
    int new_row = valid ? proposed_row : old_row;
    int new_col = valid ? proposed_col : old_col;
    int dist = abs(state->player_position[0] - old_row) + abs(state->player_position[1] - old_col);
    bool keep = alive && dist < MOB_DESPAWN_DISTANCE;
    move_mob_occupancy(state, level, old_row, old_col, new_row, new_col, keep);
    mobs->position[slot][0] = new_row;
    mobs->position[slot][1] = new_col;
    mobs->mask[slot] = keep;
}

static inline void move_ranged_slot(State* state, int level, int slot, Rng* rng) {
    Mobs* mobs = &state->ranged_mobs[level];
    bool alive = mobs->mask[slot];
    if (!alive) {
        return;
    }
    int old_row = mobs->position[slot][0];
    int old_col = mobs->position[slot][1];
    int type_id = mobs->type_id[slot];
    int cooldown = mobs->attack_cooldown[slot];

    int random_dir[2];
    choose_direction(rng_key(rng), 4, random_dir);
    int distance_row = abs(state->player_position[0] - old_row);
    int distance_col = abs(state->player_position[1] - old_col);
    int axis = choose_player_axis(rng_key(rng), distance_row, distance_col);
    int player_dir[2] = {0, 0};
    if (axis == 0) {
        player_dir[0] = signi(state->player_position[0] - old_row);
    } else {
        player_dir[1] = signi(state->player_position[1] - old_col);
    }
    int dist = distance_row + distance_col;
    int proposed_row = dist >= 6 ? old_row + player_dir[0] : old_row + random_dir[0];
    int proposed_col = dist >= 6 ? old_col + player_dir[1] : old_col + random_dir[1];
    if (dist <= 3) {
        proposed_row = old_row - player_dir[0];
        proposed_col = old_col - player_dir[1];
    }
    if (!(rng_f32(rng_key(rng)) > 0.85f)) {
        proposed_row = old_row + random_dir[0];
        proposed_col = old_col + random_dir[1];
    }
    bool valid = valid_typed_mob_position(state, level, MOB_RANGED, type_id, proposed_row, proposed_col, old_row, old_col);
    bool attacking = ((dist >= 4 && dist <= 5) || (dist <= 3 && !valid)) && cooldown <= 0 && alive;
    if (attacking) {
        static const int projectile[8] = {
            PROJECTILE_ARROW, PROJECTILE_ARROW, PROJECTILE_FIREBALL, PROJECTILE_DAGGER,
            PROJECTILE_ARROW2, PROJECTILE_SLIMEBALL, PROJECTILE_FIREBALL2, PROJECTILE_ICEBALL2
        };
        spawn_projectile(state, false, projectile[clampi(type_id, 0, 7)], old_row, old_col, player_dir[0], player_dir[1]);
        proposed_row = old_row;
        proposed_col = old_col;
    }
    int new_cooldown = attacking ? 4 : cooldown - 1;
    valid = valid_typed_mob_position(state, level, MOB_RANGED, type_id, proposed_row, proposed_col, old_row, old_col);
    int new_row = valid ? proposed_row : old_row;
    int new_col = valid ? proposed_col : old_col;
    bool keep = alive && (dist < MOB_DESPAWN_DISTANCE || fighting_boss(state));
    move_mob_occupancy(state, level, old_row, old_col, new_row, new_col, keep);
    mobs->position[slot][0] = new_row;
    mobs->position[slot][1] = new_col;
    mobs->attack_cooldown[slot] = new_cooldown;
    mobs->mask[slot] = keep;
}

static inline void update_mobs(State* state, Rng rng) {
    int level = jax_index(state->player_level, NUM_LEVELS);
    rng_key(&rng);
    move_melee_slot(state, level, 0, &rng);
    move_melee_slot(state, level, 1, &rng);
    move_melee_slot(state, level, 2, &rng);
    rng_key(&rng);
    move_passive_slot(state, level, 0, &rng);
    move_passive_slot(state, level, 1, &rng);
    move_passive_slot(state, level, 2, &rng);
    rng_key(&rng);
    move_ranged_slot(state, level, 0, &rng);
    move_ranged_slot(state, level, 1, &rng);
    rng_key(&rng);
    update_projectile_set(state, false);
    rng_key(&rng);
    update_projectile_set(state, true);
}

static inline void move_player(State* state, int action) {
    int direction[2];
    action_to_direction(action, direction);

    int proposed_row = state->player_position[0] + direction[0];
    int proposed_col = state->player_position[1] + direction[1];
    if (valid_player_position(state, proposed_row, proposed_col)) {
        state->player_position[0] = proposed_row;
        state->player_position[1] = proposed_col;
    }

    if (direction[0] != 0 || direction[1] != 0) {
        state->player_direction = action;
    }
}

static inline int level_achievement(int level) {
    switch (level) {
    case 1: return ACH_ENTER_DUNGEON;
    case 2: return ACH_ENTER_GNOMISH_MINES;
    case 3: return ACH_ENTER_SEWERS;
    case 4: return ACH_ENTER_VAULT;
    case 5: return ACH_ENTER_TROLL_MINES;
    case 6: return ACH_ENTER_FIRE_REALM;
    case 7: return ACH_ENTER_ICE_REALM;
    case 8: return ACH_ENTER_GRAVEYARD;
    default: return -1;
    }
}

static inline void change_floor(State* state, int action) {
    int level = clampi(state->player_level, 0, NUM_LEVELS - 1);
    int row = clampi(state->player_position[0], 0, MAP_SIZE - 1);
    int col = clampi(state->player_position[1], 0, MAP_SIZE - 1);

    bool on_down_ladder = state->item_map[level][row][col] == ITEM_LADDER_DOWN;
    bool can_move_down = action == ACTION_DESCEND
        && on_down_ladder
        && state->monsters_killed[level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
        && level < NUM_LEVELS - 1;

    bool on_up_ladder = state->item_map[level][row][col] == ITEM_LADDER_UP;
    bool can_move_up = action == ACTION_ASCEND
        && on_up_ladder
        && level > 0;

    if (!can_move_down && !can_move_up) {
        return;
    }

    int new_level = level + (can_move_down ? 1 : -1);
    if (can_move_down) {
        state->player_position[0] = state->up_ladders[new_level][0];
        state->player_position[1] = state->up_ladders[new_level][1];
    } else {
        state->player_position[0] = state->down_ladders[new_level][0];
        state->player_position[1] = state->down_ladders[new_level][1];
    }

    state->player_level = new_level;
    int achievement = level_achievement(new_level);
    if (achievement >= 0 && !state->achievements[achievement]) {
        state->achievements[achievement] = 1;
        state->player_xp += 1;
    }
}

static inline void level_up_attributes(State* state, int action) {
    if (state->player_xp < 1) {
        return;
    }

    bool leveled = false;
    if (action == ACTION_LEVEL_UP_DEXTERITY && state->player_dexterity < MAX_ATTRIBUTE) {
        state->player_dexterity += 1;
        leveled = true;
    } else if (action == ACTION_LEVEL_UP_STRENGTH && state->player_strength < MAX_ATTRIBUTE) {
        state->player_strength += 1;
        leveled = true;
    } else if (action == ACTION_LEVEL_UP_INTELLIGENCE && state->player_intelligence < MAX_ATTRIBUTE) {
        state->player_intelligence += 1;
        leveled = true;
    }

    if (leveled) {
        state->player_xp -= 1;
    }
}

static inline bool near_block(const State* state, int block_type) {
    static const int offsets[8][2] = {
        {0, -1}, {0, 1}, {-1, 0}, {1, 0},
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1},
    };
    int level = clampi(state->player_level, 0, NUM_LEVELS - 1);
    for (int i = 0; i < 8; i++) {
        int row = state->player_position[0] + offsets[i][0];
        int col = state->player_position[1] + offsets[i][1];
        if (row >= 0 && row < MAP_SIZE && col >= 0 && col < MAP_SIZE
            && state->map[level][row][col] == block_type) {
            return true;
        }
    }
    return false;
}

static inline int first_armour_below(const Inventory* inventory, int threshold, int* count) {
    int first = 0;
    *count = 0;
    for (int i = 0; i < 4; i++) {
        bool below = inventory->armour[i] < threshold;
        if (*count == 0 && below) {
            first = i;
        }
        *count += below ? 1 : 0;
    }
    return first;
}

static inline void craft_tools_and_items(State* state, int action) {
    bool at_table = near_block(state, BLOCK_CRAFTING_TABLE);
    bool at_furnace = near_block(state, BLOCK_FURNACE);
    Inventory* inv = &state->inventory;

    if (action == ACTION_MAKE_WOOD_PICKAXE && at_table && inv->wood >= 1 && inv->pickaxe < 1) {
        inv->wood -= 1;
        inv->pickaxe = 1;
    } else if (action == ACTION_MAKE_STONE_PICKAXE && at_table && inv->wood >= 1 && inv->stone >= 1 && inv->pickaxe < 2) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->pickaxe = 2;
    } else if (action == ACTION_MAKE_IRON_PICKAXE && at_table && at_furnace && inv->wood >= 1 && inv->stone >= 1 && inv->iron >= 1 && inv->coal >= 1 && inv->pickaxe < 3) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->iron -= 1;
        inv->coal -= 1;
        inv->pickaxe = 3;
    } else if (action == ACTION_MAKE_DIAMOND_PICKAXE && at_table && inv->wood >= 1 && inv->diamond >= 3 && inv->pickaxe < 4) {
        inv->wood -= 1;
        inv->diamond -= 3;
        inv->pickaxe = 4;
    } else if (action == ACTION_MAKE_WOOD_SWORD && at_table && inv->wood >= 1 && inv->sword < 1) {
        inv->wood -= 1;
        inv->sword = 1;
    } else if (action == ACTION_MAKE_STONE_SWORD && at_table && inv->wood >= 1 && inv->stone >= 1 && inv->sword < 2) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->sword = 2;
    } else if (action == ACTION_MAKE_IRON_SWORD && at_table && at_furnace && inv->wood >= 1 && inv->stone >= 1 && inv->iron >= 1 && inv->coal >= 1 && inv->sword < 3) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->iron -= 1;
        inv->coal -= 1;
        inv->sword = 3;
    } else if (action == ACTION_MAKE_DIAMOND_SWORD && at_table && inv->wood >= 1 && inv->diamond >= 2 && inv->sword < 4) {
        inv->wood -= 1;
        inv->diamond -= 2;
        inv->sword = 4;
    } else if (action == ACTION_MAKE_ARROW && at_table && inv->wood >= 1 && inv->stone >= 1 && inv->arrows < 99) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->arrows += 2;
    } else if (action == ACTION_MAKE_TORCH && at_table && inv->wood >= 1 && inv->coal >= 1 && inv->torches < 99) {
        inv->wood -= 1;
        inv->coal -= 1;
        inv->torches += 4;
    }

    int count = 0;
    int idx = first_armour_below(inv, 1, &count);
    if (action == ACTION_MAKE_IRON_ARMOUR && at_table && at_furnace && count > 0 && inv->iron >= 3 && inv->coal >= 3) {
        inv->iron -= 3;
        inv->coal -= 3;
        inv->armour[idx] = 1;
        state->achievements[ACH_MAKE_IRON_ARMOUR] = 1;
    }

    idx = first_armour_below(inv, 2, &count);
    if (action == ACTION_MAKE_DIAMOND_ARMOUR && at_table && count > 0 && inv->diamond >= 3) {
        inv->diamond -= 3;
        inv->armour[idx] = 2;
        state->achievements[ACH_MAKE_DIAMOND_ARMOUR] = 1;
    }
}

static inline bool can_place_item_on(int block) {
    return block == BLOCK_GRASS || block == BLOCK_SAND || block == BLOCK_PATH
        || block == BLOCK_FIRE_GRASS || block == BLOCK_ICE_GRASS;
}

static inline void add_growing_plant(State* state, int row, int col) {
    for (int i = 0; i < MAX_GROWING_PLANTS; i++) {
        if (!state->growing_plants_mask[i]) {
            state->growing_plants_positions[i][0] = row;
            state->growing_plants_positions[i][1] = col;
            state->growing_plants_age[i] = 0;
            state->growing_plants_mask[i] = 1;
            return;
        }
    }
}

static inline void place_block(State* state, int action) {
    int direction[2];
    action_to_direction(state->player_direction, direction);
    int row = state->player_position[0] + direction[0];
    int col = state->player_position[1] + direction[1];
    if (row < 0 || row >= MAP_SIZE || col < 0 || col >= MAP_SIZE) {
        return;
    }

    int level = clampi(state->player_level, 0, NUM_LEVELS - 1);
    int block = state->map[level][row][col];
    bool occupied = is_solid_block(block) || state->item_map[level][row][col] != ITEM_NONE
        || mob_at(state, level, row, col);
    Inventory* inv = &state->inventory;

    if (action == ACTION_PLACE_TABLE && !occupied && inv->wood >= 2) {
        set_block(state, level, row, col, BLOCK_CRAFTING_TABLE);
        inv->wood -= 2;
        state->achievements[ACH_PLACE_TABLE] = 1;
    } else if (action == ACTION_PLACE_FURNACE && !occupied && inv->stone >= 1) {
        set_block(state, level, row, col, BLOCK_FURNACE);
        inv->stone -= 1;
        state->achievements[ACH_PLACE_FURNACE] = 1;
    } else if (action == ACTION_PLACE_STONE && (block == BLOCK_WATER || !occupied) && inv->stone >= 1) {
        set_block(state, level, row, col, BLOCK_STONE);
        inv->stone -= 1;
        state->achievements[ACH_PLACE_STONE] = 1;
    } else if (action == ACTION_PLACE_TORCH && can_place_item_on(block) && state->item_map[level][row][col] == ITEM_NONE && inv->torches >= 1) {
        state->item_map[level][row][col] = ITEM_TORCH;
        add_light(state, level, row, col);
        inv->torches -= 1;
        state->achievements[ACH_PLACE_TORCH] = 1;
    } else if (action == ACTION_PLACE_PLANT && block == BLOCK_GRASS && state->item_map[level][row][col] == ITEM_NONE && inv->sapling >= 1) {
        set_block(state, level, row, col, BLOCK_PLANT);
        inv->sapling -= 1;
        add_growing_plant(state, row, col);
        state->achievements[ACH_PLACE_PLANT] = 1;
    }
}

static inline int choose_weighted_key(Rng key, const float* weights, int count) {
    float total = 0.0f;
    for (int i = 0; i < count; i++) {
        total += weights[i];
    }
    float draw = total * (1.0f - rng_f32(key));
    float cumulative = 0.0f;
    for (int i = 0; i < count; i++) {
        cumulative += weights[i];
        if (cumulative >= draw) {
            return i;
        }
    }
    return count - 1;
}

static inline void add_chest_loot(State* state, Rng rng) {
    Inventory* inv = &state->inventory;
    rng_key(&rng);
    (void)randint_at(rng_key(&rng), 0u, 1, 6);
    bool torch = rng_f32(rng_key(&rng)) < 0.6f;
    int torches = randint_at(rng_key(&rng), 0u, 4, 8);
    bool ore = rng_f32(rng_key(&rng)) < 0.6f;
    const float ore_weights[5] = {0.3f, 0.3f, 0.15f, 0.125f, 0.125f};
    int ore_id = choose_weighted_key(rng_key(&rng), ore_weights, 5);
    Rng amount_key = rng_key(&rng);
    int coal = randint_at(amount_key, 0u, 1, 4);
    int iron = randint_at(amount_key, 0u, 1, 3);
    int diamond = randint_at(amount_key, 0u, 1, 2);
    int sapphire = randint_at(amount_key, 0u, 1, 2);
    int ruby = randint_at(amount_key, 0u, 1, 2);
    bool potion = rng_f32(rng_key(&rng)) < 0.5f;
    int potion_id = randint_at(rng_key(&rng), 0u, 0, 6);
    int potion_amount = randint_at(rng_key(&rng), 0u, 1, 3);
    bool arrows = rng_f32(rng_key(&rng)) < 0.25f;
    int arrow_amount = randint_at(rng_key(&rng), 0u, 1, 5);
    bool tool = rng_f32(rng_key(&rng)) < 0.2f;
    int tool_id = randint_at(rng_key(&rng), 0u, 0, 2);
    const float tool_weights[4] = {0.4f, 0.3f, 0.2f, 0.1f};
    int pickaxe = choose_weighted_key(rng_key(&rng), tool_weights, 4) + 1;
    int sword = choose_weighted_key(rng_key(&rng), tool_weights, 4) + 1;
    int level = jax_index(state->player_level, NUM_LEVELS);

    if (torch) inv->torches += torches;
    if (ore && ore_id == 0) inv->coal += coal;
    if (ore && ore_id == 1) inv->iron += iron;
    if (ore && ore_id == 2) inv->diamond += diamond;
    if (ore && ore_id == 3) inv->sapphire += sapphire;
    if (ore && ore_id == 4) inv->ruby += ruby;
    if (potion) inv->potions[potion_id] += potion_amount;
    if (arrows) inv->arrows += arrow_amount;
    if (tool && tool_id == 0 && pickaxe > inv->pickaxe) inv->pickaxe = pickaxe;
    if (tool && tool_id == 1 && sword > inv->sword) inv->sword = sword;
    if (state->player_level == 1 && !state->chests_opened[level]) inv->bow = 1;
    if (!state->chests_opened[level] && (state->player_level == 3 || state->player_level == 4)) {
        inv->books += 1;
    }
}

static inline void interact_facing_tile(State* state, int action, Rng rng) {
    if (action != ACTION_DO) {
        return;
    }

    int direction[2];
    action_to_direction(state->player_direction, direction);
    int row = state->player_position[0] + direction[0];
    int col = state->player_position[1] + direction[1];
    bool in_bounds = row >= 0 && row < MAP_SIZE && col >= 0 && col < MAP_SIZE;
    int level = jax_index(state->player_level, NUM_LEVELS);
    int read_row = jax_index(row, MAP_SIZE);
    int read_col = jax_index(col, MAP_SIZE);
    int block = state->map[level][read_row][read_col];
    Inventory* inv = &state->inventory;

    bool did_attack = attack_mob_at(state, level, row, col, true);
    Rng sapling_key = rng_key(&rng);
    Rng chest_key = rng_key(&rng);
    if (did_attack || !in_bounds) {
        return;
    }

    if (block == BLOCK_TREE || block == BLOCK_FIRE_TREE || block == BLOCK_ICE_SHRUB) {
        set_block(
            state,
            level,
            row,
            col,
            block == BLOCK_TREE ? BLOCK_GRASS :
                (block == BLOCK_FIRE_TREE ? BLOCK_FIRE_GRASS : BLOCK_ICE_GRASS)
        );
        inv->wood += 1;
    } else if (block == BLOCK_STONE && inv->pickaxe >= 1) {
        set_block(state, level, row, col, BLOCK_PATH);
        inv->stone += 1;
    } else if (block == BLOCK_COAL && inv->pickaxe >= 1) {
        set_block(state, level, row, col, BLOCK_PATH);
        inv->coal += 1;
    } else if (block == BLOCK_IRON && inv->pickaxe >= 2) {
        set_block(state, level, row, col, BLOCK_PATH);
        inv->iron += 1;
    } else if (block == BLOCK_DIAMOND && inv->pickaxe >= 3) {
        set_block(state, level, row, col, BLOCK_PATH);
        inv->diamond += 1;
    } else if (block == BLOCK_SAPPHIRE && inv->pickaxe >= 4) {
        set_block(state, level, row, col, BLOCK_PATH);
        inv->sapphire += 1;
    } else if (block == BLOCK_RUBY && inv->pickaxe >= 4) {
        set_block(state, level, row, col, BLOCK_PATH);
        inv->ruby += 1;
    } else if (block == BLOCK_STALAGMITE && inv->pickaxe >= 1) {
        set_block(state, level, row, col, BLOCK_PATH);
        inv->stone += 1;
    } else if (block == BLOCK_CRAFTING_TABLE || block == BLOCK_FURNACE) {
        set_block(state, level, row, col, BLOCK_PATH);
    } else if (block == BLOCK_WATER || block == BLOCK_FOUNTAIN) {
        state->player_drink = clampi(state->player_drink + 1, 0, max_drink(state));
        state->player_thirst = 0.0f;
        state->achievements[ACH_COLLECT_DRINK] = 1;
    } else if (block == BLOCK_RIPE_PLANT) {
        set_block(state, level, row, col, BLOCK_PLANT);
        for (int i = 0; i < MAX_GROWING_PLANTS; i++) {
            if (state->growing_plants_positions[i][0] == row
                && state->growing_plants_positions[i][1] == col) {
                state->growing_plants_age[i] = 0;
                break;
            }
        }
        state->player_food = clampi(state->player_food + 4, 0, max_food(state));
        state->player_hunger = 0.0f;
        state->achievements[ACH_EAT_PLANT] = 1;
    } else if (block == BLOCK_CHEST) {
        set_block(state, level, row, col, BLOCK_PATH);
        add_chest_loot(state, chest_key);
        state->achievements[ACH_OPEN_CHEST] = 1;
    } else if (block == BLOCK_NECROMANCER && boss_vulnerable(state) && fighting_boss(state)) {
        state->boss_progress += 1;
        state->boss_timestep_to_spawn_this_round = BOSS_SPAWN_TURNS;
        state->achievements[ACH_DAMAGE_NECROMANCER] = 1;
    }
    if (block == BLOCK_GRASS && rng_f32(sapling_key) < 0.1f) inv->sapling += 1;
    state->chests_opened[level] |= block == BLOCK_CHEST;
}

static inline void drink_potion(State* state, int action) {
    int potion = action - ACTION_DRINK_POTION_RED;
    if (potion < 0 || potion >= NUM_POTIONS || state->inventory.potions[potion] <= 0) {
        return;
    }
    int effect = state->potion_mapping[potion];
    state->inventory.potions[potion] -= 1;
    if (effect == 0) state->player_health += 8.0f;
    else if (effect == 1) state->player_health -= 3.0f;
    else if (effect == 2) state->player_mana += 8;
    else if (effect == 3) state->player_mana -= 3;
    else if (effect == 4) state->player_energy += 8;
    else state->player_energy -= 3;
    state->achievements[ACH_DRINK_POTION] = 1;
}

static inline void read_book(State* state, int action, Rng rng) {
    bool reading = action == ACTION_READ_BOOK && state->inventory.books > 0;
    Rng unused;
    Rng choice_key;
    rng_split(rng, &unused, &choice_key);
    float p0 = state->learned_spells[0] ? 0.0f : 1.0f;
    float p1 = state->learned_spells[1] ? 0.0f : 1.0f;
    int spell = 0;
    if (p0 + p1 != 0.0f) {
        float r = 1.0f - rng_f32(choice_key);
        spell = r <= (p0 / (p0 + p1)) ? 0 : 1;
    }
    if (reading) {
        state->inventory.books -= 1;
        state->learned_spells[spell] = 1;
        state->achievements[spell == 0 ? ACH_LEARN_FIREBALL : ACH_LEARN_ICEBALL] = 1;
    }
}

static inline void enchant_items(State* state, int action, Rng rng) {
    int direction[2];
    action_to_direction(state->player_direction, direction);
    int level = jax_index(state->player_level, NUM_LEVELS);
    int row = jax_index(state->player_position[0] + direction[0], MAP_SIZE);
    int col = jax_index(state->player_position[1] + direction[1], MAP_SIZE);
    int block = state->map[level][row][col];
    int enchant = block == BLOCK_ENCHANTMENT_TABLE_FIRE ? 1 :
        (block == BLOCK_ENCHANTMENT_TABLE_ICE ? 2 : 0);
    int gems = enchant == 1 ? state->inventory.ruby : state->inventory.sapphire;
    bool could = state->player_mana >= 9 && enchant != 0 && gems >= 1;
    bool enchanting_sword = could && action == ACTION_ENCHANT_SWORD && state->inventory.sword > 0;
    bool enchanting_bow = could && action == ACTION_ENCHANT_BOW && state->inventory.bow > 0;
    int armour_count = 0;
    for (int i = 0; i < 4; i++) {
        armour_count += state->inventory.armour[i];
    }
    bool enchanting_armour = could && action == ACTION_ENCHANT_ARMOUR && armour_count > 0;
    Rng armour_key = rng_key(&rng);
    int unenchanted = 0;
    for (int i = 0; i < 4; i++) {
        unenchanted += state->armour_enchantments[i] == 0;
    }
    float candidates[4];
    for (int i = 0; i < 4; i++) {
        bool opposite = state->armour_enchantments[i] != 0 && state->armour_enchantments[i] != enchant;
        candidates[i] = (state->armour_enchantments[i] == 0 || (unenchanted == 0 && opposite)) ? 1.0f : 0.0f;
    }
    int armour_target = choose_weighted_key(armour_key, candidates, 4);
    if (enchanting_sword) {
        state->sword_enchantment = enchant;
        state->achievements[ACH_ENCHANT_SWORD] = 1;
    }
    if (enchanting_bow) {
        state->bow_enchantment = enchant;
    }
    if (enchanting_armour) {
        state->armour_enchantments[armour_target] = enchant;
        state->achievements[ACH_ENCHANT_ARMOUR] = 1;
    }
    bool enchanting = enchanting_sword || enchanting_bow || enchanting_armour;
    if (enchanting) {
        if (enchant == 1) {
            state->inventory.ruby -= 1;
        } else {
            state->inventory.sapphire -= 1;
        }
        state->player_mana -= 9;
    }
}

static inline void use_projectile_or_spell(State* state, int action) {
    int direction[2];
    action_to_direction(state->player_direction, direction);
    if (direction[0] == 0 && direction[1] == 0) {
        direction[0] = 1;
    }

    if (action == ACTION_SHOOT_ARROW && state->inventory.bow > 0 && state->inventory.arrows > 0) {
        bool fired = spawn_projectile(
            state,
            true,
            PROJECTILE_ARROW2,
            state->player_position[0],
            state->player_position[1],
            direction[0],
            direction[1]
        );
        if (fired) {
            state->inventory.arrows -= 1;
            state->achievements[ACH_FIRE_BOW] = 1;
        }
    } else if (action == ACTION_CAST_FIREBALL && state->learned_spells[0] && state->player_mana >= 2) {
        bool cast = spawn_projectile(
            state,
            true,
            PROJECTILE_FIREBALL,
            state->player_position[0],
            state->player_position[1],
            direction[0],
            direction[1]
        );
        if (cast) {
            state->player_mana -= 2;
            state->achievements[ACH_CAST_FIREBALL] = 1;
        }
    } else if (action == ACTION_CAST_ICEBALL && state->learned_spells[1] && state->player_mana >= 2) {
        bool cast = spawn_projectile(
            state,
            true,
            PROJECTILE_ICEBALL,
            state->player_position[0],
            state->player_position[1],
            direction[0],
            direction[1]
        );
        if (cast) {
            state->player_mana -= 2;
            state->achievements[ACH_CAST_ICEBALL] = 1;
        }
    }
}

static inline void update_plants(State* state) {
    for (int plant = 0; plant < MAX_GROWING_PLANTS; plant++) {
        if (!state->growing_plants_mask[plant]) {
            continue;
        }

        state->growing_plants_age[plant] += 1;
        if (state->growing_plants_age[plant] < 600) {
            continue;
        }

        int row = jax_index(state->growing_plants_positions[plant][0], MAP_SIZE);
        int col = jax_index(state->growing_plants_positions[plant][1], MAP_SIZE);
        if (state->growing_plants_age[plant] >= 600) {
            set_block(state, 0, row, col, BLOCK_RIPE_PLANT);
        }
    }
}

static inline void update_intrinsics(State* state, int action) {
    bool start_sleep = action == ACTION_SLEEP && state->player_energy < max_energy(state);
    state->is_sleeping = state->is_sleeping || start_sleep;

    bool wake_from_sleep = state->is_sleeping && state->player_energy >= max_energy(state);
    state->is_sleeping = state->is_sleeping && !wake_from_sleep;
    state->achievements[ACH_WAKE_UP] = state->achievements[ACH_WAKE_UP] || wake_from_sleep;

    bool start_rest = action == ACTION_REST && state->player_health < (float)max_health(state);
    state->is_resting = state->is_resting || start_rest;

    bool wake_from_rest = state->is_resting && (
        state->player_health >= (float)max_health(state)
        || state->player_food <= 0
        || state->player_drink <= 0
    );
    state->is_resting = state->is_resting && !wake_from_rest;

    bool not_boss = !fighting_boss(state);
    float decay = 1.0f - 0.125f * (float)(state->player_dexterity - 1);

    state->player_hunger += (state->is_sleeping ? 0.5f : 1.0f) * decay;
    if (state->player_hunger > 25.0f) {
        state->player_hunger = 0.0f;
        state->player_food = clampi(state->player_food - (not_boss ? 1 : 0), 0, max_food(state));
    }

    state->player_thirst += (state->is_sleeping ? 0.5f : 1.0f) * decay;
    if (state->player_thirst > 20.0f) {
        state->player_thirst = 0.0f;
        state->player_drink = clampi(state->player_drink - (not_boss ? 1 : 0), 0, max_drink(state));
    }

    if (state->is_sleeping) {
        state->player_fatigue = state->player_fatigue - 1.0f;
        if (state->player_fatigue > 0.0f) {
            state->player_fatigue = 0.0f;
        }
    } else {
        state->player_fatigue += decay;
    }
    if (state->player_fatigue > 30.0f) {
        state->player_fatigue = 0.0f;
        state->player_energy = clampi(state->player_energy - (not_boss ? 1 : 0), 0, max_energy(state));
    } else if (state->player_fatigue < -10.0f) {
        state->player_fatigue = 0.0f;
        state->player_energy = clampi(state->player_energy + 1, 0, max_energy(state));
    }

    bool all_necessities = state->player_food > 0
        && state->player_drink > 0
        && (state->player_energy > 0 || state->is_sleeping);
    state->player_recover += all_necessities
        ? (state->is_sleeping ? 2.0f : 1.0f)
        : (state->is_sleeping ? -0.5f : -1.0f) * (not_boss ? 1.0f : 0.0f);

    if (state->player_recover > 25.0f) {
        state->player_recover = 0.0f;
        state->player_health = clampf(state->player_health + 1.0f, 0.0f, (float)max_health(state));
    } else if (state->player_recover < -15.0f) {
        state->player_recover = 0.0f;
        state->player_health -= 1.0f;
    }

    float mana_gain = state->is_sleeping ? 2.0f : 1.0f;
    float mana_coeff = 1.0f + 0.25f * (float)(state->player_intelligence - 1);
    state->player_recover_mana = (state->player_recover_mana + mana_gain) * mana_coeff;
    if (state->player_recover_mana > 30.0f) {
        state->player_recover_mana = 0.0f;
        state->player_mana = clampi(state->player_mana + 1, 0, max_mana(state));
    }
}

static inline void clip_inventory_and_intrinsics(State* state) {
    state->inventory.wood = clampi(state->inventory.wood, 0, 99);
    state->inventory.stone = clampi(state->inventory.stone, 0, 99);
    state->inventory.coal = clampi(state->inventory.coal, 0, 99);
    state->inventory.iron = clampi(state->inventory.iron, 0, 99);
    state->inventory.diamond = clampi(state->inventory.diamond, 0, 99);
    state->inventory.sapling = clampi(state->inventory.sapling, 0, 99);
    state->inventory.pickaxe = clampi(state->inventory.pickaxe, 0, 99);
    state->inventory.sword = clampi(state->inventory.sword, 0, 99);
    state->inventory.bow = clampi(state->inventory.bow, 0, 99);
    state->inventory.arrows = clampi(state->inventory.arrows, 0, 99);
    state->inventory.torches = clampi(state->inventory.torches, 0, 99);
    state->inventory.ruby = clampi(state->inventory.ruby, 0, 99);
    state->inventory.sapphire = clampi(state->inventory.sapphire, 0, 99);
    state->inventory.books = clampi(state->inventory.books, 0, 99);
    for (int i = 0; i < 4; i++) {
        state->inventory.armour[i] = clampi(state->inventory.armour[i], 0, 99);
    }
    for (int i = 0; i < NUM_POTIONS; i++) {
        state->inventory.potions[i] = clampi(state->inventory.potions[i], 0, 99);
    }

    state->player_health = clampf(state->player_health, 0.0f, (float)max_health(state));
    state->player_food = clampi(state->player_food, 0, max_food(state));
    state->player_drink = clampi(state->player_drink, 0, max_drink(state));
    state->player_energy = clampi(state->player_energy, 0, max_energy(state));
    state->player_mana = clampi(state->player_mana, 0, max_mana(state));
}

static inline void calculate_inventory_achievements(State* state) {
    state->achievements[ACH_COLLECT_WOOD] |= state->inventory.wood > 0;
    state->achievements[ACH_COLLECT_STONE] |= state->inventory.stone > 0;
    state->achievements[ACH_COLLECT_COAL] |= state->inventory.coal > 0;
    state->achievements[ACH_COLLECT_IRON] |= state->inventory.iron > 0;
    state->achievements[ACH_COLLECT_DIAMOND] |= state->inventory.diamond > 0;
    state->achievements[ACH_COLLECT_SAPPHIRE] |= state->inventory.sapphire > 0;
    state->achievements[ACH_COLLECT_RUBY] |= state->inventory.ruby > 0;
    state->achievements[ACH_COLLECT_SAPLING] |= state->inventory.sapling > 0;
    state->achievements[ACH_FIND_BOW] |= state->inventory.bow > 0;
    state->achievements[ACH_MAKE_ARROW] |= state->inventory.arrows > 0;
    state->achievements[ACH_MAKE_TORCH] |= state->inventory.torches > 0;
    state->achievements[ACH_MAKE_WOOD_PICKAXE] |= state->inventory.pickaxe >= 1;
    state->achievements[ACH_MAKE_STONE_PICKAXE] |= state->inventory.pickaxe >= 2;
    state->achievements[ACH_MAKE_IRON_PICKAXE] |= state->inventory.pickaxe >= 3;
    state->achievements[ACH_MAKE_DIAMOND_PICKAXE] |= state->inventory.pickaxe >= 4;
    state->achievements[ACH_MAKE_WOOD_SWORD] |= state->inventory.sword >= 1;
    state->achievements[ACH_MAKE_STONE_SWORD] |= state->inventory.sword >= 2;
    state->achievements[ACH_MAKE_IRON_SWORD] |= state->inventory.sword >= 3;
    state->achievements[ACH_MAKE_DIAMOND_SWORD] |= state->inventory.sword >= 4;
}

static inline void update_boss_logic(State* state) {
    state->achievements[ACH_DEFEAT_NECROMANCER] |= has_beaten_boss(state);
    if (fighting_boss(state)) {
        state->boss_timestep_to_spawn_this_round -= 1;
    }
}

static inline float calculate_light_level(int timestep) {
    float progress = fmodf((float)timestep / (float)DAY_LENGTH, 1.0f) + 0.3f;
    float cosine = cosf(3.14159265358979323846f * progress);
    return 1.0f - powf(fabsf(cosine), 3.0f);
}

static inline bool is_game_over(const State* state) {
    return state->player_health <= 0.0f || state->timestep >= DEFAULT_MAX_TIMESTEPS;
}
