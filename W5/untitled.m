clear; clc;

function [r_values, g_values] = calculate_radial_distribution(positions, dr, rho, L)
    N = size(positions, 1);
    max_bin_index = floor(L / 2 / dr);
    g_values = zeros(1, max_bin_index);

    for i = 1:N
        for j = i+1:N
            r = distance(positions(i,:), positions(j,:), L);
            if r < L / 2
                bin_index = min(floor(r / dr), max_bin_index);
                g_values(bin_index) = g_values(bin_index) + 2; % Count each pair only once
            end
        end
    end

    % Normalize g(r)
    for i = 1:length(g_values)
        r_lower = (i - 1) * dr;
        r_upper = i * dr;
        shell_volume = pi * (r_upper^2 - r_lower^2);
        g_values(i) = g_values(i) / (shell_volume * rho * N);
    end

    r_values = dr:dr:(max_bin_index * dr);
end


% Definizione delle costanti
epsi = 0.997;
sigma = 3.405;

% Potenziale di Lennard-Jones
lj_potential = @(r) 4 * epsi * ((sigma ./ r) .^ 12 - (sigma ./ r) .^ 6);

% Funzione per calcolare la distanza tra due punti con condizioni periodiche
distance = @(r1, r2, L) sqrt(mod((r1(1) - r2(1)), L).^2 + mod((r1(2) - r2(2)), L).^2);

% Parametri
N = 242; % Numero di atomi
T = 0.5; % Temperatura
p = 0.96; % Densità
L = sqrt(N / p); % Dimensione della scatola
D = 0.15; % Amplitudine dello spostamento
n_steps = 100000; % Numero di passaggi

% Inizializzazione delle posizioni casuali
positions = rand(N, 2) * L;

% Inizializzazione dell'energia
energy = 0.0;

Tstart = tic;

for step = 1:n_steps
    % Seleziona casualmente un atomo
    atom_index = randi(N);

    % Calcola l'energia prima dello spostamento
    old_energy = physconst('Boltzmann') * T;
    for i = 1:N
        if i ~= atom_index
            old_energy = old_energy + lj_potential(distance(positions(atom_index,:), positions(i,:), L));
        end
    end

    % Propone uno spostamento casuale
    displacement = (rand(1, 2) - 0.5) * D;
    new_position = positions(atom_index,:) + displacement;
    if any(new_position < 0) || any(new_position >= L)
        continue;
    end

    % Applica le condizioni periodiche al nuovo punto
    new_position = mod(new_position, L);

    % Calcola l'energia dopo lo spostamento
    new_energy = physconst('Boltzmann') * T;
    for i = 1:N
        if i ~= atom_index
            new_energy = new_energy + lj_potential(distance(new_position, positions(i,:), L));
        end
    end

    % Criterio di Metropolis
    delta_energy = new_energy - old_energy;
    if delta_energy < 0 || rand() < exp(-delta_energy / T)
        positions(atom_index,:) = new_position;
    end
end

disp(['Tempo di esecuzione: ', num2str(toc(Tstart))]);

% Salva su un singolo file
dlmwrite(['configurazione_', datestr(now, 'yyyymmdd-HHMMSS'), '.txt'], positions);
dlmwrite(['parametri_', datestr(now, 'yyyymmdd-HHMMSS'), '.txt'], [N, T, p, D, n_steps, epsi, sigma, toc(Tstart)]);

figure()
plot(positions(:,1), positions(:,2), 'o');
axis([0 L 0 L]);
title('Final configuration');
xlabel('x');
ylabel('y');
grid on;

% Inizializzazione degli array per i valori di energia
U_values = zeros(1, N*2);

% Calcolo dei valori di energia
for i = 1:N
    for j = i+1:N
        r = distance(positions(i,:), positions(j,:), L);
        U_values(i) = U_values(i) + lj_potential(r);
        U_values(j) = U_values(j) + lj_potential(r);
    end
end

% Calcolo dell'energia media e della deviazione standard
E_avg = mean(U_values);
E_std = std(U_values);

disp(['Average energy: ', num2str(E_avg)]);
disp(['Energy standard deviation: ', num2str(E_std)]);

[r_values, g_values] = calculate_radial_distribution(positions, 0.1, p, L);

figure()
plot(r_values, g_values);
title('Radial distribution function');
xlabel('r');
ylabel('g(r)');
grid on;
