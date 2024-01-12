//ea3.js

/**********************event listeners************************************************/
//submit chat button
document.getElementById('send-button').addEventListener('click', sendMessage);

//submit chat via Enter (not Shift+Enter)
document.getElementById('chat-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

//prediction in der Konsole
document.getElementById('chat-input').addEventListener('keyup', function(e) {
    if (e.key === ' ') {
        let predictions = predictRNN(e.target.value);
        console.log(predictions);
    }
});

/*************************sending a message into chat**********************************/
function sendMessage() {
    var input = document.getElementById('chat-input');
    var chatSpacing = document.getElementById('chat-spacing');

    // Only send the message if the input is not empty
    if (input.value.trim() !== '') {
        // Create a new chat-history element
        var newChatHistory = document.createElement('div');
        newChatHistory.id = 'chat-history';
        newChatHistory.innerHTML = '<p>' + input.value + '</p>';

        // Add the new chat-history to the beginning of the chat-spacing
        chatSpacing.prepend(newChatHistory);

        // Scroll to the bottom of the chat-spacing
        chatSpacing.scrollTop = chatSpacing.scrollHeight;

        // Clear the input field
        input.value = '';

        // Set focus back to the input field
        input.focus();
    }
}


/*********************************************************************** begin: models ************************************************/
//ea3.js
console.log(`Die TensorFlow.js-Version ist ${tf.version.tfjs}`);

let textData = loadTextData(); //textdaten: oliver twist (von unten)

// Textbereinigung
textData = textData.trim().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,""); //Sonderzeichen
textData = textData.replace(/[\r\n]+/g, ' '); //Zeilenumbr�che
console.log('Bereinigte Textdaten:', textData);

// Tokenisierung
function tokenizer(text) {
    return text.split(' ');
}
let tokens = tokenizer(textData);
tokens = tokens.filter(token => token !== ''); //leere Tokens entfernen
console.log('Tokenisierung abgeschlossen.');
console.log('Tokenisierte Daten:', tokens);

//Datenmenge reduzieren
tokens = tokens.slice(0, 10);
console.log('Reduzierte tokenisierte Daten:', tokens);

// Erstellen Sie ein W�rterbuch
let wordIndex = {};
tokens.forEach((token, i) => {
    if (!(token in wordIndex)) {
        wordIndex[token] = i + 1;
    }
});
console.log('W�rterbuch:', wordIndex);
console.log('W�rterbuch erstellt.');

//vocabSize
let vocabSize = Object.keys(wordIndex).length;
console.log('unique words:', vocabSize);

// Sequenzbildung
let sequences = tokens.map(token => wordIndex[token]);
console.log('Sequenzen:', sequences);
let xs = [];
let ys = [];
for (let i = 0; i < sequences.length - 1; i++) {
    xs.push(sequences[i]);
    ys.push(sequences[i + 1]);
}
console.log('xs:', xs);
console.log('ys:', ys);
console.log('Sequenzbildung und One-Hot-Encoding abgeschlossen.');
console.log('Daten sind bereit f�r das Modell.');
console.log('L�nge von xs:', xs.length);

// Umwandlung in One-Hot-Vektoren
let xsOneHot = tf.oneHot(tf.tensor1d(xs, 'int32'), vocabSize);
let ysOneHot = tf.oneHot(tf.tensor1d(ys, 'int32'), vocabSize);

// Hinzuf�gen einer Dimension f�r Zeitschritte - wie viele W�rter sich angeschaut werden in der Sequenz, um das n�chste vorherzusagen
xsOneHot = xsOneHot.reshape([xsOneHot.shape[0], 1, xsOneHot.shape[1]]);


/***************************************************************************** RNN ******************************************************************************/


// Modell erstellen
const RNNmodel = tf.sequential();
RNNmodel.add(tf.layers.simpleRNN({units: 10, inputShape: [1, vocabSize]}));
RNNmodel.add(tf.layers.dense({units: vocabSize, activation: 'softmax'}));
console.log('RNN Modell erstellt');

// Modell kompilieren
RNNmodel.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
console.log('RNN Modell kompiliert');

// Modell trainieren
const batchSize = 128;
const epochs = 10;
RNNmodel.fit(xsOneHot, ysOneHot, {batchSize, epochs}).then(() => {
    console.log('RNN Modelltraining abgeschlossen.');
});

// Vorhersagefunktion
function predictRNN(seedText) {
    let seedTokens = tokenizer(seedText);
    console.log('text zur prediction:', seedTokens);
    let seedSequences = seedTokens.map(token => wordIndex[token]);
    let seedOneHot = tf.oneHot(tf.tensor1d(seedSequences, 'int32'), vocabSize);
    seedOneHot = seedOneHot.reshape([seedOneHot.shape[0], 1, seedOneHot.shape[1]]);
    let prediction = RNNmodel.predict(seedOneHot);
    let top3Indices = tf.topk(prediction, 3).indices;
    let predictedIndices = Array.from(top3Indices.dataSync());
    return predictedIndices.map(index => Object.keys(wordIndex).find(key => wordIndex[key] === index));
}

/***************************************************************************** FFNN *****************************************************************************/

// Umwandlung in One-Hot-Vektoren
xsOneHot = tf.oneHot(tf.tensor1d(xs, 'int32'), vocabSize);
ysOneHot = tf.oneHot(tf.tensor1d(ys, 'int32'), vocabSize);

// Modell erstellen
const FFNNmodel = tf.sequential();
FFNNmodel.add(tf.layers.dense({units: 10, inputShape: [vocabSize], activation: 'relu'}));
FFNNmodel.add(tf.layers.dense({units: vocabSize, activation: 'softmax'}));
console.log('FFNN Modell erstellt');

// Modell kompilieren
FFNNmodel.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
console.log('FFNN Modell kompiliert');

// Modell trainieren
const batchSizeFFNN = 128;
const epochsFFNN = 10;
FFNNmodel.fit(xsOneHot, ysOneHot, {batchSize: batchSizeFFNN, epochs: epochsFFNN}).then(() => {
    console.log('FFNN Modelltraining abgeschlossen.');
});

// Vorhersagefunktion
function predictFFNN(seedText) {
    let seedTokens = tokenizer(seedText);
    let seedSequences = seedTokens.map(token => wordIndex[token]);
    let seedOneHot = tf.oneHot(tf.tensor1d(seedSequences, 'int32'), vocabSize);
    let prediction = FFNNmodel.predict(seedOneHot);
    let predictedIndex = tf.argMax(prediction, 1).dataSync()[0];
    return Object.keys(wordIndex).find(key => wordIndex[key] === predictedIndex);
}

/***************************************************************************** text laden ***********************************************************************/
function loadTextData() {
    let textData = `

                             Oliver Twist

                                  von

                            Charles Dickens

                  mit einer biographischen Einleitung
                          von Johannes Gaulke

                     Globus Verlag G.m.b.H. Berlin




    Charles Dickens


    Unter den gro�en Humoristen des vorigen Jahrhunderts, die zugleich
    Tendenzschriftsteller im besten Sinne waren, nimmt Charles Dickens
    einen hervorragenden Platz ein, den er trotz des schnellen Wandels des
    literarischen Geschmacks und der Kunstanschauung in der Weltliteratur
    behaupten wird. Dickens ist nicht nur der Lieblingsdichter seines
    Volkes, sondern er ist schon zu Lebzeiten in allen L�ndern des
    Erdenrunds heimisch geworden. In H�tte und Palast sind seine Werke
    gedrungen und haben �berall starke und nachhaltige Wirkungen ausge�bt.
    Begabt mit dem k�stlichen Humor, der mit dem einen Auge weint und dem
    anderen lacht, ist Dickens allen denen, die auf der H�he des Lebens
    wandeln, ein treuer Mentor geworden, den Elenden und Enterbten des
    Lebensgl�cks aber ein aufrichtiger Freund und Tr�ster.

    Charles Dickens konnte zum ganzen Volke von allen den Dingen, die
    unsre Welt ausmachen, sprechen, weil er das Leben gr�ndlich kannte,
    weil er selbst alle Wechself�lle des Lebens an sich selbst erfahren
    hatte. Als Kind wenig bemittelter Eltern am 7. Februar 1812 in Landport
    bei Portsmouth geboren, mu�te er schon im Alter von zehn Jahren,
    als sein Vater in London ins Schuldgef�ngnis gewandert war, f�r den
    eigenen Lebensunterhalt sorgen. W�hrend er als Laufbursche gegen einen
    k�rglichen Wochenlohn t�tig war, vernachl�ssigte er naturgem�� seine
    Schulbildung g�nzlich, und er geno� erst, nachdem der Vater eine
    bescheidene Stellung in London erlangt hatte, als zw�lfj�hriger Knabe
    einen besseren Unterricht. Den Mangel eines systematischen Unterrichts
    hat er durch Selbstunterricht, der sich auf alle Gebiete des
    Wissens erstreckte, namentlich aber durch das Studium der englischen
    Schriftsteller ausgeglichen. Im Jahre 1833 ver�ffentlichte er, nachdem
    er sich schon als Journalist an f�hrenden Bl�ttern unter dem Pseudonym
    Boz mit gro�em Erfolge bet�tigt hatte, sein erstes Buch, eine Reihe von
    Skizzen aus dem Londoner Volksleben in zwei B�nden. Einige Jahre sp�ter
    folgten die �_Pickwick papers_�, die ihn mit einem Schlage zu einem
    gelesenen und in allen Schichten gleich gesch�tzten Autor machten.
    Das Buch, das in einer Reihe von lose aneinandergef�gten Skizzen die
    Abenteuer einiger Mitglieder des Pickwickklubs auf ihrer Reise durch
    England schildert, enth�lt in gewissem Sinne das Programm des sp�teren
    Dickens, der das Leben schildert, wie es sich ihm darbietet, immer
    von dem Gedanken getragen, moralische Wirkungen zu erzielen und den
    Menschen mit seiner Umwelt zu vers�hnen. Um dieses Ziel zu erreichen,
    schrickt er nicht vor �bertreibungen eines Zustandes oder einer
    Handlung zur�ck und macht selbst, um m�glichst eindringlich zu wirken,
    seine Figuren, die meistens sehr lebensvoll einsetzen, zu menschlichen
    Karikaturen.

    In rascher Folge erscheinen in den drei�iger und vierziger Jahren
    des vorigen Jahrhunderts die Hauptwerke Dickens. Die Reihe er�ffnet
    �Oliver Twist� (1838), das als das erste realistische, aus dem
    Volkstum gesch�pfte Buch mit au�erordentlichem Enthusiasmus in England
    aufgenommen wurde und bald seinen Weg �ber den Erdball machte.
    Es folgten: �Nicholas Nickleby� (1839) und �_Master Humphrey's
    clock_� (1840), ein Werk, das sich �hnlich wie die �Pickwickier�
    aus Einzelerz�hlungen zusammensetzt, sich aber vor einem ernsteren
    Hintergrund abspielt und tiefergreifende Menschenschicksale darstellt.

    In den vierziger Jahren unternahm Dickens, der inzwischen zu einem
    gewissen Wohlstand gelangt war, gro�e Auslandsreisen. Die Hauptfrucht
    seiner ersten Amerikareise (1842) ist der Roman �Martin Chuzzlewit�,
    in dem er die Heuchelei der Amerikaner mit scharfen Hieben gei�elt.
    Auch in seinen �_American notes_� l��t er es an harten Bemerkungen
    �ber die Amerikaner und amerikanischen Einrichtungen nicht fehlen. Die
    Amerikaner haben ihm die geringe Meinung �ber sie und ihr Land, der
    er zu wiederholten Malen Ausdruck gegeben hat, nicht nachgetragen,
    sondern ihm in Neuyork, Chicago und anderen St�dten pr�chtige Denkm�ler
    errichtet.

    In Italien schrieb Dickens den Roman �Chimes� (1844), am Genfer
    See �_Battle of Life_� (1846). Fast gleichzeitig entstand �_Dombey
    and son_�, ein Lebensbild aus dem B�rgertum, in dem Episoden von
    ergreifender Tragik und grotesker Komik einander folgen. Auf der H�he
    des Schaffens stehend, schrieb Dickens Ende der vierziger Jahre den
    autobiographischen Roman �David Copperfield�, der nach Plan und Anlage
    als ein wahrhaft geniales Werk genannt zu werden verdient. In der
    Charakterisierung der Person hat Dickens hier die h�chste Meisterschaft
    erreicht, auch ist die Handlung einheitlicher und geschlossener als in
    den Werken seiner ersten Periode.

    David Copperfield ist wie die meisten Romane ein sozialer Tendenzroman.
    F�r Dickens, der aus dem Volke hervorgegangen war, der auch als
    Dichter ein Selfmademan war, war die Kunst immer nur ein Mittel zum
    Zweck, nicht Selbstzweck, wie es eine sp�tere franz�sische Richtung
    durch den Grundsatz �_l'art pour l'art_� ausdr�ckt. Dickens ist
    daher keiner begrenzten Gruppe oder Kunstrichtung einzureihen; er
    ist weder Realist noch Idealist im herk�mmlichen Sinne, sondern auch
    als K�nstler immer nur Moralist. Zwar sind die Zust�nde stets mit
    den Augen des Realisten gesehen, er ist sogar ein Kleinmaler von
    einer Pr�gnanz des Ausdrucks wie wenige, aber dar�ber hinaus reicht
    sein Wirklichkeitssinn nicht. Sobald er an den Menschen herantritt,
    versagt sein Charakterisierungsverm�gen, er schildert die Menschen
    nicht wie sie sind, aus dem Milieu heraus, sondern wie er w�nscht, da�
    sie sein m�chten. Nur selten gelingt es ihm, einen der Wirklichkeit
    entsprechenden Menschen zu zeichnen; seine Romanfiguren sind entweder
    idealisiert oder karikiert -- im besten Falle Typen, keine Individuen.
    Entweder sind sie Erzb�sewichter oder herzensgute Engel. Und zum Schlu�
    erhalten sie alle, ganz im Einklang mit dem h�chsten moralischen
    Grundgesetz, ihre Strafe oder ihre Belohnung f�r das, was sie getan
    oder unterlassen haben.

    Am besten gelingen Dickens die Gestalten aus dem Volk, mit ihnen ist
    der Dichter aufgewachsen, mit ihnen hat er gelitten, mit ihnen kann er
    daher auch empfinden. Auch in die Seele des Kindes vermag sich Dickens
    zu versetzen; hier wirkt sein Pathos immer echt, ob er das Elend des
    ausgesetzten Kindes schildert, die Qualen und Entbehrungen eines
    kleinen Bettlers oder gar den Tod eines ungl�cklichen kleinen Wesens.

    Je weiter sich Dickens vom Volkstum entfernt, umso unklarer und
    verschwommener werden seine Gestalten, doch wei� er auch hier
    wiederum mit gl�cklichem Griff das Milieu, in dem eine Lordschaft
    oder gar ein englischer Herzog sich bewegt, festzuhalten. Man
    sieht gern �ber die angedeuteten Schw�chen hinweg, da der Dichter
    unersch�pflich in der Erfindung komischer und grotesker Situationen
    ist und mit einem von Herzen kommenden und zu Herzen gehenden Humor
    alle menschlichen Schw�chen und Verirrungen zu entschuldigen wei�.
    Selbst dem tiefgesunkenen Verbrecher haftet immer noch ein menschlich
    liebensw�rdiger Zug an. Ohne gerade Kriminalpsychologe zu sein,
    schildert Dickens seine Gestalten fast durchg�ngig als Produkte ihrer
    Umgebung und behandelt auch den sch�ndlichsten Misset�ter mit Nachsicht
    und Milde. So nur konnte er zu einem Anwalt der Ungl�cklichen und
    Enterbten werden.

    In der zweiten Periode seines dichterischen Schaffens, die die beiden
    letzten Jahrzehnte seines Lebens umfa�t, treten die Eigenarten und
    Schw�chen des Dichters immer sch�rfer hervor. Rastlos t�tig, lockert
    sich in seinen Romanen immer mehr die Komposition, auf langatmige
    Schilderungen folgen knappe dramatische Evolutionen und spannende
    Konflikte, die zu einem pl�tzlichen Abschlu� dr�ngen. Besonders
    charakteristisch ist in dieser Beziehung der vierb�ndige Roman �_Our
    mutual friend_�, aber auch �_Bleakhouse_� und �_Tale of two cities_�,
    wo die franz�sische Revolution den Hintergrund bildet, lassen die
    Einheitlichkeit des Plans vermissen.

    Charles Dickens war w�hrend seines ganzen Lebens von einem
    Arbeitseifer, der weder Rast noch Ruh kennt, beseelt. W�hrend er seine
    gro�en Romane schrieb, war er im Nebenfach als Journalist und Redakteur
    t�tig. Im Jahre 1845 trat er in die Redaktion der neubegr�ndeten
    Zeitung �_Daily News_�, die auch seine italienischen Reisebilder zuerst
    ver�ffentlichte, ein. 1849 gab er eine Wochenschrift �_Household
    Words_�, die der Unterhaltung und Belehrung diente, heraus. Daneben
    fand er Zeit zu Vortragsreisen in England, Irland und Amerika, die ihm
    Reicht�mer und hohe Ehrungen einbrachten, aber auch die mittelbare
    Ursache zu seinem pl�tzlichen Tode wurden. Er starb, vom Schlage
    getroffen, nach kurzem Krankenlager auf seinem Landsitz Gadshill,
    am 9. Juli 1870 im Alter von 58 Jahren. Seine Gebeine wurden in der
    Westminsterabtei, dem Pantheon Englands, beigesetzt.

    Wenden wir uns nunmehr der in diesem Bande ver�ffentlichten Erz�hlung
    �Oliver Twist� zu, so werden wir die Vorz�ge und Schw�chen Dickensscher
    Erz�hlungskunst gerade an diesem Werke h�chst eindringlich wahrnehmen
    k�nnen. Oliver Twist ist Dickens hervorragendstes Jugendwerk und
    behandelt die Geschichte einer Jugend. Zweifellos haben eigene
    Jugendeindr�cke dem Dichter die Direktive zu dieser Arbeit gegeben.
    Wie der kleine Oliver, so hat auch Dickens, zwar unter anderen
    Verh�ltnissen, aber ebenso m�hselig, sich emporringen m�ssen. Das
    Leben hatte den Dichter schon in zarter Jugend hart angepackt, aber
    wie das Gold sich im Feuer l�utert, so l�utert sich die Seele im
    Lebenskampf, der Schmutz haftet nur dem Schmutzigen an, wer gesund und
    rein empfindet, mu� schlie�lich alle Widerw�rtigkeiten des Lebens
    �berwinden. Das ist der Leitgedanke in Oliver Twist. H�chst drastische
    Bilder l��t der Dichter vor unserem geistigen Auge entstehen, scharf
    zugespitzte Situationen schildert er mit einer Anschaulichkeit, die uns
    um das Schicksal des jugendlichen Helden mit banger Sorge erf�llt. Wir
    empfinden und f�hlen mit Oliver Twist, wir f�rchten gar um sein Leben
    und zittern um sein Seelenheil. Oftmals hat es den Anschein, als m�sse
    die Katastrophe j�h �ber ihn hereinbrechen, aber immer wieder entwirren
    sich die verworrenen Schicksalsf�den, bis ihm endlich die Erl�sung aus
    unw�rdigen Zust�nden, in die er ohne seine Schuld geraten ist, wird.

    Wenn man den moralischen Ma�stab an eine dichterische Arbeit anlegen
    will, so vollzieht sich in �Oliver Twist� alles ganz folgerichtig:
    die Tugend mu� schlie�lich siegen, denn so will es die moralische
    Weltordnung. Vom literarischen Gesichtspunkt betrachtet, lie�e sich
    allerdings mancherlei gegen den Optimismus Dickens einwenden; man
    merkt gar zu schnell die moralisierende Absicht und wird verstimmt.
    Dagegen kann Dickens als Zustandsschilderer auch hier vor jeder
    literarischen Kritik bestehen. Wie anschaulich sind allein die
    Verbrecherschlupfwinkel geschildert! Wie �berzeugend die �rtlichkeiten
    des dunkelsten Londons! Man gewinnt hier �berall den Eindruck des
    Selbstgeschauten. Dickens bedient sich zur Erreichung seines Zwecks
    oft ungew�hnlicher Mittel und verbl�ffender Wendungen. Er konstruiert
    die unwahrscheinlichsten Situationen und nimmt es auch mit den
    Tatsachen nicht so genau, um eine Kontrastwirkung zu erzielen. Einzelne
    Begebenheiten streifen fast das Niveau des Kolportageromans, w�hrend
    andere den Eindruck h�chster K�nstlerschaft auf den Leser machen.

    �Oliver Twist� ist eine Arbeit, die nicht mit dem Kopf, sondern mit
    dem Herzen geschrieben ist. Es ist der Roman des Kindes, vielleicht
    der erste dieser Art in der neueren Literatur. Der ma�lose Jammer der
    ausgesetzten und verlassenen Kinder, von denen es im heutigen London
    noch hunderte und aberhunderte gibt, hat den Dichter angeregt zu einer
    Arbeit, die ein Appell an die Welt zur Abhilfe der verrotteten Zust�nde
    sein soll. Wir leben im �Jahrhundert des Kindes�! M�nner und Frauen
    aller Kreise haben zusammengewirkt, um eine Hebung des sittlichen
    Niveaus, aber auch der materiellen Lage der Kinder der �rmsten zu
    erzielen. Der Dichter des �Oliver Twist� verdient als Vorl�ufer dieser
    Bewegung bezeichnet zu werden. Menscheng�te und Kinderliebe sprechen
    aus jeder Zeile des Buches; ohne diese Qualit�ten h�tte es schwerlich
    seinen Platz in der Weltliteratur behauptet.

    Von dem feinen Verst�ndnis der Kinderseele und den Bed�rfnissen des
    Kindes legt neben �Oliver Twist� auch Dickens �_A child's history of
    England_� ein gl�nzendes Zeugnis ab. In einer, den Anschauungskreis des
    Kindes angemessenen Form schildert Dickens hier die Hauptereignisse der
    englischen Geschichte mit h�chster Eindringlichkeit und Wahrhaftigkeit,
    aber auch zugleich frei von jeder aufdringlichen chauvinistischen
    Tendenz. Das Buch, das in England und Amerika zu den meistgelesenen
    B�chern z�hlt, hat bei uns bei weitem nicht die ihm geb�hrende
    Beachtung gefunden, obgleich wir in der geschichtlichen Jugendliteratur
    ihm nichts Ebenb�rtiges zur Seite stellen k�nnen.

    Charles Dickens hat in England viele Nachahmer gefunden, aber niemand
    hat ihn, dem das Schreiben eine sittliche Aufgabe war, auch nur im
    entferntesten erreicht. Seine Nachahmer haben ihn eigentlich nur
    in der Breite der Anlage und der Umst�ndlichkeit der Schilderung
    getroffen, nicht aber in der Eindringlichkeit seines Vortrags und
    in der �berzeugungskraft, mit der er seine Tendenz verficht. Bald
    wird ein halbes Jahrhundert seit seinem Tode verflossen sein, die
    englische Familienblattliteratur hat inzwischen hunderttausende von
    Neuerscheinungen auf den Markt geworfen, aber doch ist Dickens der
    Dichter des Volkes geblieben. Von den gro�en Humoristen des vorigen
    Jahrhunderts kann ihm nur einer als gleichwertig zur Seite gestellt
    werden: unser Fritz Reuter, der ja auch wie Dickens in einer harten
    Schule der Entbehrungen zum Dichter und Menschenfreund herangewachsen
    ist. Sie liebten beide das Volk, weil sie es als echte S�hne des Volkes
    genau kannten, und sie haben beide dadurch, da� sie die erb�rmliche
    Allt�glichkeit mit echter Poesie und echtem Humor durchtr�nkt haben,
    Millionen entz�ckt und mit dem Leben vers�hnt. Wer das zuwege bringt,
    ist ein Wohlt�ter der Menschheit. In diesem Sinne hat sich einer der
    gr��ten Staatsm�nner Englands, Gladstone, �ber Dickens ge�u�ert; auch
    er feierte ihn als einen Erzieher und Wohlt�ter seines Volkes. Charles
    Dickens wird daher, mag er einer strengen literarischen Kritik auch
    nicht immer standhalten, mag er selbst als Menschenschilderer �berholt
    worden sein, dennoch weiterleben und noch viele Generationen durch
    seinen k�stlichen Humor entz�cken und begeistern.

                                                           Johannes Gaulke.




    1. Kapitel.

        Handelt von dem Orte, an dem Oliver Twist geboren wurde, und von
        den seine Geburt begleitenden Umst�nden.


    Eine Stadt, die ich aus gewissen Gr�nden nicht n�her bezeichnen will,
    der ich aber auch keinen erdichteten Namen beilegen m�chte, besitzt
    unter anderen �ffentlichen Geb�uden gleich den meisten anderen St�dten,
    sie m�gen gro� oder klein sein, von alters her ein Armenhaus, und in
    diesem wurde an einem Tage, dessen genaues Datum f�r den Leser kein
    besonderes Interesse hat, das Mitglied der sterblichen Menschheit
    geboren, dessen Name in der �berschrift dieses Kapitels angegeben ist.

    Lange Zeit, nachdem der Wundarzt des Kirchspiels ihn in diese Welt
    der M�hen und Sorgen bef�rdert hatte, blieb es �u�erst zweifelhaft,
    ob er lange genug leben w�rde, um �berhaupt eines Namens zu bed�rfen.
    Es war n�mlich tats�chlich mit erheblichen Schwierigkeiten verbunden,
    Oliver dahin zu bringen, da� er sich der Aufgabe, Atem zu holen,
    selbst unterzog -- einem m�hsamen Gesch�fte, das die Gewohnheit uns
    aber freilich zu einer notwendigen Lebensbedingung gemacht hat; eine
    Zeitlang lag er nach Luft schnappend auf einer kleinen Matratze
    aus Schafwolle und schien sich in der Schwebe zwischen dieser und
    jener Welt zu befinden, wobei die Wage sich entschieden zugunsten
    der letzteren neigte. Wenn Oliver w�hrend dieser kurzen Zeit von
    sorglichen Gro�m�ttern, gesch�ftigen Tanten, erfahrenen W�rterinnen
    und hochgelahrten Doktoren umgeben gewesen w�re, so w�rde er nat�rlich
    die Stunde nicht �berlebt haben, allein es war niemand in seiner
    N�he, au�er einer alten Frau, die sich infolge des ungewohnten
    Genusses von Bier in einer etwas angeheiterten Stimmung befand, und
    dem Kirchspielwundarzte, der die Geburtshilfe kontraktm��ig leistete.
    Oliver und die Natur fochten also die Sache zwischen sich ganz
    allein aus, und die Folge davon war, da� nach kurzem Kampfe Oliver
    atmete, nieste und endlich den Insassen des Armenhauses die Tatsache
    ank�ndigte, da� dem Kirchspiele eine neue Last aufgeb�rdet worden sei,
    indem er ein so lautes Geschrei erhob, wie man es f�glicherweise von
    einem neugeborenen Knaben erwarten konnte.

    Als Oliver diesen ersten Beweis von der freien und selbst�ndigen
    T�tigkeit seiner Lungen gab, bewegte sich die geflickte Decke, die
    nachl�ssig �ber die eiserne Bettstelle gebreitet war; das bleiche
    Antlitz einer jungen Frau erhob sich matt von dem harten Pf�hle, und
    eine schwache Stimme brachte m�hsam die Worte hervor: �Lassen Sie mich
    das Kind sehen, dann will ich gern sterben.�

    Der Wundarzt, der vor dem Kamine sa� und seine H�nde abwechselnd an dem
    Feuer w�rmte und rieb, erhob sich bei den Worten der jungen Frau, trat
    an das Kopfende des Bettes und sagte mit mehr Freundlichkeit im Tone,
    als man ihm zugetraut haben w�rde: �Oh, Sie d�rfen jetzt nicht vom
    Sterben sprechen.�

    �Der Herr segne ihr gutes Herzchen, nein!� unterbrach ihn die W�rterin,
    indem sie eine gr�ne Glasflasche, von deren Inhalt sie in einer
    verschwiegenen Ecke mit sichtlichem Behagen gekostet hatte, rasch in
    die Tasche steckte. �Der Herr segne ihr gutes Herzchen; wenn sie erst
    so alt geworden ist wie ich und dreizehn Kinder gehabt hat und alle
    sind tot bis auf zwei, die zusammen mit mir im Armenhause sind, so wird
    sie schon auf andere und vern�nftigere Gedanken kommen; der Herr segne
    ihr gutes Herzchen. Bedenken Sie nur, Frauchen, was es hei�t, Mutter
    eines so s��en, kleinen L�mmchens zu sein.�

    Diese tr�stlichen Worte schienen ihre Wirkung zu verfehlen. Die
    W�chnerin sch�ttelte den Kopf und streckte die Arme nach dem Kinde
    aus. Der Wundarzt reichte es ihr, sie k��te es, heftig erregt, mit
    den kalten, wei�en Lippen auf die Stirn, fuhr mit den H�nden �ber ihr
    Gesicht, blickte wild umher, schauderte, sank zur�ck -- und starb.

    �'s ist aus mit ihr�, sagte der Wundarzt nach einigen Bem�hungen, sie
    wieder zum Leben zur�ckzubringen.

    �Das arme Kind!� sagte die W�rterin, indem sie den Pfropfen der
    gr�nen Flasche aufhob, der auf das Kissen gefallen war, als sie sich
    niederbeugte, um das Kind aufzunehmen. �Armes Kind!�

    �Sie brauchen nicht zu mir zu schicken, wenn das Kind schreit�, fuhr
    der Wundarzt fort, w�hrend er kaltbl�tig seine Handschuhe anzog. �Es
    wird wahrscheinlich sehr unruhig sein; geben Sie ihm dann ein wenig
    Hafergr�tze.� Er setzte den Hut auf, trat noch einmal an das Bett und
    sagte: �Die Mutter sah gut aus; woher kam sie?�

    �Sie wurde gestern abend gebracht,� erwiderte die W�rterin, �auf Befehl
    des Direktors. Man hatte sie auf der Stra�e liegend gefunden, und
    sie mu� ziemlich weit hergewandert sein, denn ihre Schuhe waren ganz
    zerrissen; aber woher sie kam, oder wohin sie wollte, das wei� niemand.�

    Der Wundarzt beugte sich �ber die Verblichene, hob die rechte Hand
    derselben empor und bemerkte kopfsch�ttelnd: �Die alte Geschichte; kein
    Trauring, wie ich sehe. Hm! gute Nacht!�

    Er ging zu seinem Abendessen, und die W�rterin setzte sich, nachdem
    sie sich noch einmal an der gr�nen Flasche erlabt hatte, auf einen
    Stuhl in der N�he des Feuers und begann das Kind anzukleiden.
    Bis zu diesem Augenblick h�tte man nicht sagen k�nnen, ob es das
    Kind eines Edelmannes oder eines Bettlers sei; das d�rftige,
    verwaschene Kinderzeug des Armenhauses bezeichnete indes sogleich
    seine gegenw�rtige und zuk�nftige Stellung in der Welt, sein ganzes
    Schicksal, als Kirchspielkind -- Waise des Armenhauses, halb verhungert
    und unter M�he und Plackerei, verachtet von allen, bemitleidet von
    niemand, durch die Welt geknufft und gesto�en zu werden.

    Oliver schrie mit kr�ftiger Stimme; h�tte er wissen k�nnen, da� er eine
    Waise war, �berliefert der z�rtlichen F�rsorge von Kirchen�ltesten und
    Kirchenvorstehern, so w�rde er vielleicht noch lauter geschrien haben.




    2. Kapitel.

        Handelt von Oliver Twists Heranwachsen und k�mmerlicher Ern�hrung
        sowie von einer Sitzung des Armenkollegiums.


    W�hrend der n�chsten acht bis zehn Monate war Oliver das Opfer einer
    systematischen Gaunerei und Betr�gerei. Er wurde aufgep�ppelt. Die
    elende und verlassene Lage der kleinen Waise wurde von der Beh�rde
    des Armenhauses pflichtschuldigst der des Kirchspiels gemeldet. Die
    letztere forderte von der ersteren w�rdevoll einen Bericht dar�ber ab,
    ob sich nicht in �dem Hause� eine Frauensperson bef�nde, die dem Kinde
    seine nat�rliche Nahrung reichen k�nnte. Die Beh�rde des Armenhauses
    beantwortete die Anfrage untert�nigst mit nein, und daraufhin fa�te
    die Kirchspielbeh�rde den hochherzigen Entschlu�, Oliver in ein etwa
    drei Meilen entferntes Filialarmenhaus bringen zu lassen, wo zwanzig
    bis drei�ig andere kleine �bertreter der Armengesetze unter der
    m�tterlichen Aufsicht einer �ltlichen Frau, welche f�r jeden derselben
    w�chentlich sieben und einen halben Penny erhielt, aufwuchsen, ohne
    zu gut gen�hrt oder zu warm gekleidet und verz�rtelt zu werden. Mit
    sieben und einem halben Penny l��t sich nicht viel beschaffen, und die
    Matrone war klug und erfahren. Sie wu�te, wie leicht sich Kinder den
    Magen �berladen k�nnen und was ihnen dient, ebenso genau aber auch, was
    ihr selbst gut war; sie verwendete daher einen betr�chtlichen Teil des
    f�r die Kinder Bestimmten in ihrem eigenen Nutzen, fand demnach in der
    tiefsten noch eine tiefere Tiefe und bewies somit, da� sie es in der
    Experimentalphilosophie wirklich weit gebracht hatte.

    Jedermann kennt die Geschichte eines anderen Experimentalphilosophen,
    nach dessen ruhmw�rdiger Theorie ein Pferd imstande war, ohne Nahrung
    zu leben, und der jene so vortrefflich demonstrierte, da� er sein
    eigenes Pferd bis auf einen Strohhalm den Tag herunterbrachte, und ohne
    Frage ein �u�erst mutiges, kr�ftiges und gar nicht fressendes Tier
    aus ihm gemacht haben w�rde, wenn es nicht vierundzwanzig Stunden vor
    seinem ersten komfortablen vollkommenen Hungertage gestorben w�re. Die
    mehrerw�hnte Matrone wendete dasselbe System nicht selten mit gleichem
    Ungl�cke auf die Kirchspielkinder an, deren nicht wenige vor K�lte oder
    Hunger, oder weil sie einen Fall getan oder sich verbrannt hatten,
    starben und zu ihren V�tern in jener Welt, die sie in dieser nicht
    gekannt, versammelt wurden, wenn sie sie eben mit vieler M�he so weit
    gebracht hatte, da� sie von der m�glichst geringen Quantit�t schwacher
    Nahrungsmittel leben konnten.

    Stellten die Direktoren unangenehme Untersuchungen �ber den Verbleib
    eines Kindes an, oder taten die Geschworenen l�stige Fragen, so
    sch�tzten das Zeugnis und die Aussage des Wundarztes und des
    Kirchspieldieners gegen diese Zudringlichkeiten. Der erstere hatte
    stets die Leichen ge�ffnet und nichts darin gefunden (was sehr
    nat�rlich zuging), und der letztere beschwor stets, was dem Kirchspiel
    angenehm war, und gab damit einen gro�en Beweis von Selbstaufopferung
    und Hingebung. Das Armenkollegium besuchte von Zeit zu Zeit die
    Filialanstalt und schickte tags zuvor den Kirchspieldiener, um seine
    Ankunft zu verk�nden. Und dann sahen die Kinder stets gut und reinlich
    aus, und was konnte man mehr verlangen?

    Es war nicht zu verlangen, da� die in der Filiale herrschende
    Hausordnung ein allzu �ppiges Gedeihen der Kinder bef�rderte, und
    so war auch Oliver Twist an seinem neunten Geburtstage ein blasses,
    schw�chliches, im Wachstum zur�ckgebliebenes Kind von sehr geringem
    Leibesumfange; doch wohnte in ihm ein gesunder, kr�ftiger Geist, der
    auch, dank der strengen Di�t des Hauses, hinreichenden Raum hatte, sich
    auszudehnen. Oliver feierte seinen Geburtstag im Kohlenkeller in der
    erlesenen Gesellschaft zweier anderer junger Herren, die nach einer
    t�chtigen Tracht Schl�ge hier mit ihm eingesperrt worden waren, weil
    sie sich erk�hnt hatten, hungrig zu sein, als Frau Mann, die gutherzige
    Pflegerin, durch die Erscheinung Mr. Bumbles, des Kirchspieldieners,
    der dem Gartenpf�rtchen zuschritt, in Schrecken gesetzt wurde.

    �Du meine G�te, sind Sie es, Mr. Bumble?� rief sie ihm aus dem Fenster,
    anscheinend hoch erfreut, entgegen. -- �Susanne, bring gleich den
    Oliver und die anderen beiden Buben herauf und wasch sie. Ach, Mr.
    Bumble, wie lange haben Sie sich nicht sehen lassen!�

    Mr. Bumble war ein wohlbeleibter und dazu cholerischer Mann, und so
    r�ttelte er, anstatt auf diese freundliche Begr��ung in h�flicher Weise
    zu antworten, w�tend an der kleinen Pforte und gab ihr dann einen Sto�,
    wie ihn nur ein Kirchspieldiener versetzen konnte.

    �Herr des Himmels!� rief Mrs. Mann, indem sie aus dem Zimmer st�rzte --
    denn die drei Knaben waren inzwischen entfernt worden --, �da� ich es
    auch dieser lieben Kinder wegen vergessen mu�te, da� die T�r von innen
    verriegelt ist. Treten Sie ein, Sir, bitte, treten Sie ein, Mr. Bumble!
    Haben Sie die G�te.�

    Obgleich diese Einladung von einem freundlichen L�cheln begleitet
    war, das sogar das Herz eines Kirchen�ltesten erweicht haben w�rde,
    bes�nftigte es den Kirchspieldiener doch keineswegs.

    �Nennen Sie das einen respektvollen oder schicklichen Empfang, Mrs.
    Mann,� fragte Bumble, indem er seinen Stab fester in die Hand nahm,
    �wenn Sie die Kirchspielbeamten an Ihrer Gartenpforte warten lassen,
    wenn sie in Parochialangelegenheiten in betreff der Parochialkinder
    hierher kommen?�

    �Ich kann Ihnen versichern, Mr. Bumble, da� ich nur ein paar der lieben
    Kinder bei mir hatte, wegen deren Sie so freundlich sind, herzukommen�,
    erwiderte Mrs. Mann mit gro�er Unterw�rfigkeit.

    Mr. Bumble hegte eine hohe Meinung von seiner oratorischen Begabung und
    seiner Wichtigkeit. Er hatte die eine bewiesen und die andere gewahrt.
    Er war in milderer Stimmung.

    �Nun, nun, Mrs. Mann,� sagte er, �es mag sein, wie Sie sagen, es mag
    sein. Lassen Sie mich hinein, Mrs. Mann; ich komme in Gesch�ften und
    habe Ihnen etwas zu sagen.�

    Mrs. Mann n�tigte den Kirchspieldiener in ein kleines Sprechzimmer,
    bot ihm einen Stuhl an und legte dienstbeflissen seinen dreieckigen
    Hut und seinen Stab auf den Tisch vor ihm. Mr. Bumble wischte sich den
    Schwei� von der Stirn, blickte freundlich auf den dreieckigen Hut und
    l�chelte. Ja, er l�chelte. Kirchspieldiener sind auch nur Menschen, und
    Mr. Bumble l�chelte.

    �Nehmen Sie es mir nicht �bel, was ich Ihnen sagen will�, bemerkte Mrs.
    Mann mit bezaubernder Liebensw�rdigkeit. �Sie wissen, Sie haben einen
    weiten Weg hinter sich; wollen Sie nicht ein Gl�schen nehmen?�

    �Nicht einen Tropfen, nicht einen Tropfen�, versetzte Mr. Bumble, indem
    er mit seiner rechten Hand in w�rdevoller, aber freundlicher Weise
    abwinkte.

    �Ich denke, Sie werden mir schon den Gefallen tun�, sagte Mrs. Mann,
    die den Ton der Weigerung und die diese begleitende Geb�rde bemerkt
    hatte. �Nur ein ganz kleines Gl�schen mit einem Schluck kalten Wassers
    und einem St�ck Zucker.�

    Mr. Bumble hustete.

    �Nur ein ganz kleines Gl�schen�, wiederholte Mrs. Mann in dringendem
    Tone.

    �Was ist es denn?� fragte der Kirchspieldiener.

    �Nun, es ist das, von dem ich etwas im Hause zu halten verpflichtet
    bin, um es den lieben Kindern in den Kaffee gie�en zu k�nnen, wenn
    sie nicht wohl sind, Mr. Bumble�, entgegnete Mrs. Mann, w�hrend sie
    ein Eckschr�nkchen �ffnete und eine Flasche und ein Glas herausnahm.
    �Es ist Genever, ich will Sie nicht hintergehen, Mr. Bumble. Es ist
    Genever.�

    �Geben Sie den Kindern Kaffee, Mrs. Mann?� fragte Bumble, der mit
    seinen Augen den interessanten Vorgang der Mischung verfolgte.

    �Ach, gesegne es ihnen Gott, ich tue es, so kostspielig es auch sein
    mag�, versetzte die W�rterin. �Ich k�nnte sie vor meinen leiblichen
    Augen nicht leiden sehen, Sir, Sie wissen es ja.�

    �Nein�, sagte Mr. Bumble beistimmend; �nein, Sie k�nnten es nicht. Sie
    sind eine menschlich denkende Frau, Mrs. Mann.� (Hier setzte sie das
    Glas vor ihn hin.) �Ich werde so bald wie m�glich Gelegenheit nehmen,
    es dem Kollegium gegen�ber zu erw�hnen, Mrs. Mann.� (Er zog das Glas
    n�her zu sich heran.) �Sie empfinden wie eine Mutter.� (Er ergriff
    das Glas.) �Ich -- ich trinke mit Vergn�gen auf Ihre Gesundheit, Mrs.
    Mann�, und er trank es zur H�lfte aus.

    �Und nun zu den Gesch�ften!� rief der Kirchspieldiener, indem er eine
    lederne Brieftasche hervorzog. �Der Knabe, der halb auf den Namen
    Oliver Twist getauft wurde, ist heute neun Jahre alt.�

    �Des Himmels Segen �ber das liebe Herzchen!� rief Mrs. Mann aus und
    mu�te die Augen mit der Sch�rze abtrocknen.

    Mr. Bumble fuhr fort: �Trotz ausgebotener Belohnung von zehn Pfund, ja
    nachher von zwanzig Pfund -- trotz der �bernat�rlichen Anstrengungen
    des Kirchspiels, sind wir nicht imstande gewesen, seinen Vater
    ausfindig zu machen oder seiner Mutter Wohnung, Namen oder Stand in
    Erfahrung zu bringen.�

    �Wie geht es denn aber zu, da� er einen Namen hat?� fragte die
    Waisenmutter.

    Der Kirchspieldiener warf sich in die Brust und erwiderte: �Ich erfand
    ihn.�

    �Sie, Mr. Bumble!�

    �Ich, Mrs. Mann. Wir benennen unsere Findlinge nach dem Alphabet. Der
    letzte war ein S -- Swubble: ich benannte ihn. Dieser war ein T --
    Twist: ich gab ihm abermals den Namen. Der n�chste, der kommen wird,
    wird Unwin hei�en, der n�chstfolgende Vilkins. Ich habe Namen im Vorrat
    von A bis Z; und wenn ich beim Z angekommen bin, fang' ich beim A
    wieder an.�

    �Sie sind wirklich ein Gelehrter, Mr. Bumble!�

    �Mag sein, mag sein, Mrs. Mann. Doch genug davon. Oliver ist jetzt
    zu alt geworden zum Hierbleiben, das Kollegium hat beschlossen, ihn
    zur�ckzunehmen, ich bin selbst gekommen, ihn abzuholen; -- wo ist er?�

    Mrs. Mann eilte hinaus und erschien gleich darauf mit Oliver wieder,
    der unterdes gewaschen und bestens gekleidet war.

    �Mach 'nen Diener vor dem Herrn, Oliver�, sagte sie.

    Oliver verbeugte sich tief vor dem Kirchspieldiener auf dem Stuhle und
    dem dreieckigen Hute auf dem Tische.

    �Willst du mit mir gehen, Oliver?� redete ihn Mr. Bumble in feierlichem
    Tone an.

    Oliver war im Begriff, zu antworten, da� er auf das bereitwilligste mit
    jedermann fortgehen w�rde, hob aber zuf�llig die Augen zu Mrs. Mann
    empor, die hinter des Kirchspieldieners Stuhl getreten war und mit
    grimmigen Mienen die Faust sch�ttelte. Er wu�te nur zu gut, was das
    bedeutete.

    �Geht *sie* auch mit?� fragte er.

    �Das ist unm�glich; sie wird aber bisweilen kommen und dich besuchen�,
    erwiderte Bumble.

    Das war kein gro�er Trost f�r Oliver; allein er hatte trotz seiner
    Jugend Verstand genug, sich anzustellen, als verlie�e er das Haus nur
    sehr ungern; ohnehin standen ihm die Tr�nen infolge des Hungers und
    soeben erfahrener harter Z�chtigung nahe genug. Mrs. Mann umarmte ihn
    wiederholt und gab ihm, was er am meisten bedurfte, ein gro�es St�ck
    Butterbrot, damit er im Armenhause nicht zu hungrig anlangte. Die
    Sache war nat�rlich abgemacht. Sein Butterbrot in der Hand, verlie�
    er die St�tte, wo kein Strahl eines freundlichen Blickes das Dunkel
    seiner ersten Kinderjahre erhellt hatte. Und doch brach er in Tr�nen
    kindlichen Schmerzes aus, als das Gartentor sich hinter ihm schlo�.
    Verlie� er doch seine Leidensgef�hrten, die einzigen Freunde, die er in
    seinem Leben gekannt hatte; und zum erstenmal seit dem Erwachen seines
    Bewu�tseins empfand er ein Gef�hl seiner Verlassenheit in der gro�en,
    weiten Welt. Mr. Bumble schritt kr�ftig vorw�rts; der kleine Oliver
    trabte neben ihm her und fragte am Ende jeder Meile, ob sie nicht bald
    �da� sein w�rden. Auf diese Fragen gab Mr. Bumble sehr kurze und
    m�rrische Antworten; denn die zeitweilige Milde, die der Genu� von
    Genever und Wasser in manchen Gem�tern erzeugt, war l�ngst verflogen,
    und er war wiederum Kirchspieldiener.

    Oliver war noch nicht eine Viertelstunde innerhalb der Mauern des
    Armenhauses gewesen und hatte kaum ein zweites St�ck Brot vertilgt,
    als Mr. Bumble, der ihn der Obhut einer alten Frau �bergeben hatte,
    zur�ckkehrte. Er erkl�rte ihm, da� heute abend eine Sitzung des
    Armenkollegiums stattf�nde, und da� er sofort vor diesem zu erscheinen
    habe.

    Oliver, der keine allzu klare Vorstellung von dem hatte, was ein
    Armenkollegium zu bedeuten habe, war von dieser Mitteilung wie bet�ubt
    und wu�te nicht, ob er lachen oder weinen sollte. Er hatte jedoch keine
    Zeit, �ber diesen Punkt nachzudenken; denn Mr. Bumble versetzte ihm mit
    seinem Stabe einen Schlag auf den Kopf, um ihn aufzuwecken, und einen
    anderen �ber den R�cken, um ihn munter zu machen. Dann befahl er ihm,
    ihm zu folgen, und f�hrte ihn in ein gro�es, wei�get�nchtes Zimmer, in
    dem acht bis zehn wohlbeleibte Herren um einen Tisch herumsa�en. Oben
    am Tische sa� in einem Armstuhl, der h�her war als die �brigen, ein
    besonders wohlgen�hrter Herr mit einem sehr runden, roten Gesichte.

    �Mache dem Kollegium eine Verbeugung�, sagte Bumble. Oliver zerdr�ckte
    zwei oder drei Tr�nen in seinen Augen, und da er kein Kollegium,
    sondern nur den Tisch sah, so machte er vor diesem eine wohlgelungene
    Verbeugung.

    �Wie hei�t du, Junge?� begann der Herr auf dem gro�en Stuhle.

    Oliver zitterte, denn der Anblick so vieler Herren brachte ihn g�nzlich
    au�er Fassung; Bumble suchte ihn durch eine kr�ftige Ber�hrung mit
    dem Kirchspieldienerstabe zu beleben, und er fing an zu weinen. Er
    antwortete daher leise und z�gernd, worauf ihm ein Herr in wei�er Weste
    zurief, er w�re ein dummer Junge, was ein vortreffliches Mittel war,
    ihm Mut einzufl��en.

    �Junge,� sagte der Pr�sident, �h�re, was ich dir sage. Du wei�t doch,
    da� du eine Waise bist?�

    �Was ist denn das, Sir?� fragte der ungl�ckliche Oliver.

    �Er ist in der Tat ein dummer Junge -- ich sah es gleich�, sagte der
    Herr mit der wei�en Weste sehr bestimmt.

    �Du wirst doch wissen,� nahm der Herr wieder das Wort, der zuerst
    gesprochen hatte, �da� du weder Vater noch Mutter hast und vom
    Kirchspiel erzogen wirst?�

    �Ja, Sir�, antwortete Oliver, bitterlich weinend.

    �Was heulst du?� fragte der Herr mit der wei�en Weste; und es war in
    der Tat h�chst auffallend, da� Oliver weinte. Was konnte er denn f�r
    eine Veranlassung dazu haben?

    �Ich hoffe doch, da� du jeden Abend dein Gebet hersagst�, fiel ein
    anderer Herr in barschem Tone ein, �und f�r diejenigen, die dir
    zu essen geben und f�r dich sorgen, betest, wie es sich f�r einen
    Christenmenschen ziemt.�

    �Ja, Sir�, stotterte Oliver.

    �Wir haben dich hierher bringen lassen,� sagte der Pr�sident, �damit du
    erzogen werden und ein n�tzliches Gesch�ft lernen sollst. Du wirst also
    morgen fr�h um sechs Uhr anfangen, Werg zu zupfen.�

    F�r die Vereinigung dieser beiden Wohltaten in der einfachen
    Besch�ftigung des Wergzupfens machte Oliver unter Nachhilfe des
    Kirchspieldieners eine tiefe Verbeugung und ward dann eiligst in einen
    gro�en Saal gef�hrt, wo er sich auf einem rauhen, harten Bette in den
    Schlaf weinte. Welch ein ehrenvolles Licht f�llt hierdurch auf die
    milden Gesetze Englands! Sie gestatten den Armen, zu schlafen!

    Armer Oliver! Als er so in gl�cklicher Unbewu�theit seiner ganzen
    Umgebung schlafend dalag, dachte er nicht daran, da� das Kollegium an
    ebendemselben Tage zu einer Entscheidung gelangt war, die den gr��ten
    Einflu� auf seine k�nftigen Geschicke aus�ben sollte. Die Sache
    verhielt sich n�mlich folgenderma�en: Die Mitglieder des Kollegiums
    waren sehr weise, den Dingen auf den Grund gehende, philosophisch
    gebildete M�nner, und als sie dazu kamen, ihre Aufmerksamkeit dem
    Armenhause zuzuwenden, fanden sie mit einem Male, was gew�hnliche
    Sterbliche niemals entdeckt h�tten. Den Armen gefiel es darin nur zu
    gut! Es war ein regelrechter Unterschlupfsort f�r die �rmeren Klassen,
    ein Gasthaus, in dem man nichts zu zahlen hatte -- ein Ort, an dem
    man das ganze Jahr hindurch auf �ffentliche Kosten das Fr�hst�ck, das
    Mittagessen, den Tee und das Abendbrot einnehmen konnte -- ein Elysium
    aus Ziegeln und M�rtel, in dem nur gescherzt und gespielt, aber nicht
    gearbeitet wurde. �Oho,� sagte das Kollegium, �wir sind die richtigen
    M�nner, um hier Ordnung zu schaffen!� So ordneten sie denn an, da� alle
    Armen die Wahl haben sollten (denn sie wollten um alles in der Welt
    niemand zwingen), langsam in oder rasch au�er dem Hause zu verhungern.
    In dieser Absicht schlossen sie mit den Wasserwerken einen Vertrag
    �ber die Lieferung einer unbegrenzten Menge Wasser und mit einem
    Getreideh�ndler einen ebensolchen �ber die in gro�en Zwischenr�umen
    erfolgenden Lieferungen von kleinen Mengen Hafermehl ab und gaben
    t�glich drei Portionen eines d�nnen Mehlbreies aus; au�erdem wurde
    zweimal w�chentlich eine Zwiebel und des Sonntags eine halbe Semmel
    gereicht.

    Die ersten sechs Monate nach der Aufnahme Oliver Twists war das System
    in vollem Gange. Das Gemach, in welchem die Knaben gespeist wurden,
    war eine Art K�che, und der Speisemeister, unterst�tzt von ein paar
    Frauen, teilte ihnen aus einem kupfernen Kessel am unteren Ende ihre
    Haferbreiportionen zu, einen Napf voll und nicht mehr, ausgenommen
    an Sonn- und Feiertagen, wo sie auch noch ein nicht eben zu gro�es
    St�ck Brot bekamen. Die N�pfe brauchten nicht gewaschen zu werden,
    denn sie wurden mit den L�ffeln der Knaben so lange poliert, bis sie
    wieder vollkommen blank waren; und auch an den L�ffeln und Fingern
    blieben Speisereste niemals h�ngen. Kinder pflegen eine vortreffliche
    E�lust zu besitzen. Oliver und seine Kameraden hatten drei Monate die
    Hungerdi�t ausgehalten, vermochten sie nun aber nicht l�nger mehr
    zu ertragen. Ein f�r sein Alter sehr gro�er Knabe, dessen Vater ein
    Garkoch gewesen, erkl�rte den �brigen, da� er, wenn er nicht t�glich
    zwei N�pfe Haferbrei bekomme, f�rchten m�sse, �ber kurz oder lang
    seinen Bettkameraden, einen kleinen, schw�chlichen Knaben, aufzuessen.
    Seine Augen waren verst�rt, und rollten wild. Die halbverhungerte Schar
    glaubte ihm, hielt einen Rat, loste darum, wer nach dem Abendessen zum
    Speisemeister gehen und um mehr bitten solle, und das Los traf Oliver
    Twist.

    Der Abend kam, der Speisemeister stellte sich an den Kessel, der
    Haferbrei wurde ausgef�llt und ein breites Gebet �ber der schmalen
    Kost gesprochen. Die letztere war verschwunden, die Knaben fl�sterten
    untereinander, winkten Oliver, und die zun�chst Sitzenden stie�en
    ihn an. Der Hunger lie� ihn alle Bedenklichkeiten und R�cksichten
    vergessen. Er stand auf, trat mit Napf und L�ffel vor den Speisemeister
    hin und sagte, freilich mit ziemlichem Beben: �Bitt' um Vergebung, Sir,
    ich m�chte noch ein wenig.�

    Der wohlgen�hrte, rotwangige Speisemeister erbla�te, starrte den
    kleinen Rebellen wie bet�ubt vor Entsetzen an und mu�te sich am Kessel
    festhalten. Die Frauen waren vor Erstaunen, die Knaben vor Schreck
    sprachlos. �Was willst du?� fragte der Speisemeister endlich mit
    schwacher Stimme. Oliver wiederholte unter Furcht und Zittern seine
    Worte, und nunmehr ermannte sich der Speisemeister, schlug ihn mit dem
    L�ffel auf den Kopf und rief laut nach dem Kirchspieldiener.

    Das Armenkollegium war eben versammelt, als Mr. Bumble in gro�er
    Erregung hereinst�rzte und, zu dem Herrn auf dem hohen Stuhle gewandt,
    sagte: �Mr. Limbkins, ich bitte um Verzeihung, Sir! Oliver Twist hat
    mehr gefordert.�

    Das Kollegium war starr. Entsetzen �ber eine solche Frechheit malte
    sich auf allen Gesichtern.

    �Mehr?� erwiderte Mr. Limbkins. �Fassen Sie sich, Bumble, und antworten
    Sie mir klar und deutlich. Verstehe ich recht, da� er mehr gefordert
    hat, nachdem er die von dem Direktorium festgesetzte Portion verzehrt
    hatte?�

    �Jawohl, Sir�, entgegnete Bumble.

    �Denken Sie an mich, Gentlemen,� sagte der Herr mit der wei�en Weste,
    �der Knabe wird dereinst geh�ngt werden.�

    Niemand widersprach dieser Prophezeiung. Es entspann sich eine lebhafte
    Diskussion. Oliver wurde auf Befehl des Kollegiums sofort eingesperrt,
    und am n�chsten Morgen wurde ein Anschlag an die Au�enseite des Tores
    geklebt, in dem jedermann, der Oliver Twist zu sich nehmen wollte, die
    Summe von f�nf Pfund zugesprochen wurde -- mit anderen Worten, man bot
    Oliver Twist um f�nf Pfund an jedermann aus, sei es Mann oder Frau,
    der einen Lehrling oder Laufburschen brauchte, gleichviel wer und in
    welchem Handwerke oder Gesch�fte.




    3. Kapitel.

        Berichtet, wie Oliver Twist nahe daran war, eine Anstellung zu
        bekommen, welche keine Sinekure gewesen w�re.


    Wenn es Oliver darum zu tun gewesen w�re, die Prophezeiungen des Herrn
    mit der wei�en Weste selbst wahr zu machen, so h�tte er zum wenigsten
    Zeit genug dazu gehabt; denn er blieb acht Tage lang eingesperrt.
    Allein, um sich im Gef�ngnis zu erh�ngen, fehlte ihm erstlich ein
    Taschentuch -- denn Taschent�cher waren als Luxusartikel verp�nt --,
    und zweitens war er noch zu sehr Kind. Er weinte daher nur den langen
    Tag �ber, und wenn die lange, grausige Nacht kam, so deckte er seine
    H�ndchen �ber seine Augen, um nicht in die Dunkelheit starren zu
    m�ssen, kroch in einen Winkel und versuchte zu schlafen. Aber immer
    und immer wieder fuhr er vor Angst und Entsetzen aus seinem unruhigen
    Schlummer empor und dr�ngte sich dichter und dichter an die Wand
    heran, als w�re selbst ihre kalte, harte Fl�che ein Schutz f�r ihn in
    der Finsternis und Einsamkeit, die ihn rings umgaben.

    Es war indes daf�r gesorgt, da� es ihm an Leibesbewegung, Gesellschaft
    und religi�sem Trost nicht mangelte.

    Was die Leibes�bungen betrifft, so war es sch�nes, kaltes Wetter,
    und er durfte seine Waschungen jeden Morgen unter der Pumpe in einem
    gepflasterten Hofe vornehmen in der Gegenwart des Herrn Bumble, der
    durch wiederholte Anwendung seines Stabes daf�r sorgte, da� er sich
    nicht erk�ltete, und da� eine prickelnde Empfindung seinen K�rper
    durchlief. Was die Gesellschaft betrifft, so wurde er jeden zweiten Tag
    in den Saal gef�hrt, wo die Knaben ihr Mittagbrot verzehrten, und wo er
    vor deren Augen zum warnenden Beispiel ausgepeitscht wurde. Und weit
    entfernt, da� ihm die Segnungen des religi�sen Zuspruchs vorenthalten
    worden w�ren, wurde er vielmehr jeden Abend zur Gebetsstunde in
    denselben Raum gesto�en; hier durfte er zuh�ren und seinem Gem�te
    Tr�stung zuf�hren, da auf Anordnung des Kollegiums ein allgemeines
    Gebet der Knaben eingef�gt worden war, das eine besondere Klausel
    enthielt, in der sie zu Gott flehten, er m�ge sie gut, tugendhaft,
    zufrieden und gehorsam machen und vor der S�ndhaftigkeit und
    Lasterhaftigkeit Oliver Twists bewahren.

    W�hrend Olivers Angelegenheiten sich in diesem vielversprechenden und
    g�nstigen Zustande befanden, ereignete es sich eines Morgens, da� der
    Schornsteinfegermeister Mr. Gamfield auf der Landstra�e langsam seines
    Weges zog, in tiefem Sinnen �ber die Mittel und Wege, wie er seine
    Miete, wegen deren er von seinem Hauswirt schon zu wiederholten Malen
    gemahnt worden war, bezahlen sollte. Mr. Gamfield mochte den Stand
    seiner Finanzen noch so sanguinisch betrachten: es fehlten ihm immer
    noch f�nf Pfund an der n�tigen Summe, und in einer Art arithmetischer
    Verzweiflung zermarterte er sein Gehirn und mi�handelte seinen Esel,
    als er, am Armenhause angelangt, den Anschlag am Tore erblickte.

    �Brrr!� sagte Mr. Gamfield zu dem Esel.

    Der Esel war ebenfalls in tiefes Nachdenken versunken und besch�ftigte
    sich wahrscheinlich gelegentlich mit der Frage, ob er einen oder zwei
    Kohlstr�nke erhalten w�rde, wenn er die beiden S�cke Ru�, mit denen der
    kleine Karren beladen war, an Ort und Stelle gebracht h�tte, und so
    trottete er denn weiter, ohne auf den Zuruf seines Herrn zu achten.

    Mr. Gamfield stie� halblaut einen schweren Fluch aus, rannte dem Esel
    nach und gab ihm einen Schlag auf den Kopf, der jeden anderen Sch�del,
    ausgenommen den eines Esels, zertr�mmert haben w�rde. Dann ergriff er
    den Z�gel und ri� scharf an dem Kinnbacken des Tieres, um ihm in zarter
    Weise zu Gem�te zu f�hren, da� er nicht sein eigener Herr sei; durch
    diese Mittel gelang es ihm, den Esel herumzulenken. Dann gab er ihm
    einen zweiten Schlag auf den Kopf, um ihn bis zu seiner R�ckkehr zu
    bet�uben, und schritt, nachdem er diese Vorsichtsma�regeln getroffen
    hatte, auf das Tor zu, um den Anschlag zu lesen.

    Der Herr mit der wei�en Weste stand, die Arme auf dem R�cken gekreuzt,
    vor dem Tore, nachdem er in dem Beratungszimmer einige tiefempfundene
    Wahrheiten zum besten gegeben hatte. Er hatte den kleinen Zwist
    zwischen Mr. Gamfield und dem Esel beobachtet und l�chelte h�chst
    vergn�gt, als der Mann n�her trat, um den Anschlag zu lesen, da er auf
    den ersten Blick sah, da� Mr. Gamfield gerade der richtige Lehrherr f�r
    Oliver sei. Auch Mr. Gamfield l�chelte, als er das Schriftst�ck las,
    denn f�nf Pfund waren gerade die Summe, die er brauchte, und was den
    Knaben betrifft, den er dazunehmen sollte, so wu�te Mr. Gamfield, dem
    es bekannt war, welcher Art die Kost im Armenhause war, da� es sich um
    einen ganz kleinen, schm�chtigen Kerl handeln w�rde, wie geschaffen f�r
    die neuen Patentschornsteine. Daher las er den Anschlag noch einmal von
    Anfang bis zu Ende durch, fa�te als Beweis f�r seine H�flichkeit an
    seine Pelzm�tze und wandte sich an den Herrn in der wei�en Weste.

    �Dieser Junge hier, den das Armenhaus als Lehrling vergeben will ...�
    begann Mr. Gamfield.

    �Ach, lieber Mann,� erwiderte der Mann in der wei�en Weste
    herablassend, �was ist mit ihm?�

    �Wenn das Kirchspiel ihn ein leichtes, angenehmes Handwerk, das
    achtungswerte Schornsteinfegerhandwerk, erlernen lassen will, so
    brauche ich einen Lehrling und bin bereit, ihn zu nehmen.�

    �Treten Sie n�her�, entgegnete der Mann in der wei�en Weste. Mr.
    Gamfield lief erst noch einmal zur�ck, um dem Esel noch einen Schlag
    vor den Kopf zu versetzen und am Zaume zu rei�en, als Warnung, er
    m�ge es sich nicht etwa einfallen lassen, in seiner Abwesenheit
    durchzugehen, und folgte dann dem Herrn mit der wei�en Weste in das
    Zimmer, wo Oliver diesen zuerst gesehen hatte.

    �Es ist ein schmutziges Gewerbe�, erwiderte Mr. Limbkins, als Mr.
    Gamfield seinen Wunsch abermals vorgebracht hatte.

    �Es ist auch schon vorgekommen, da� Knaben in den Schornsteinen
    erstickt sind�, sagte ein anderer Herr.

    �Das kam nur daher,� versetzte Gamfield, �da� man das Stroh na� machte,
    ehe man es im Kamin anz�ndete, um die Jungen herunterzuholen; es gab
    nur Rauch, aber keine Flamme. Rauch aber ist ganz unzweckm��ig, um
    einen Jungen herunterzuholen, denn er veranla�t ihn nur zum Schlafen,
    und das eben ist es, was er will. Jungens sind widerspenstig und faul,
    meine Herren, und ein gutes Feuer ist das beste Mittel, sie rasch zum
    Herunterkommen zu bringen. Es ist auch ein ganz humanes Mittel, denn
    wenn sie in der Esse steckengeblieben sind, so arbeiten sie, wenn sie
    sich die F��e verbrennen, aus Leibeskr�ften, sich loszumachen.�

    Der Herr in der wei�en Weste schien sich �ber diese Erkl�rung h�chlich
    zu belustigen, aber seine Heiterkeit wurde durch einen strafenden
    Blick, den ihm Mr. Limbkins zuwarf, sofort ged�mpft. Die Direktoren
    berieten nun ein paar Minuten miteinander, aber in so leisem Tone, da�
    nur die Worte �Ersparnis� und �guten Eindruck bei der Abrechnung�, die
    mit gro�em Nachdruck mehrmals wiederholt wurden, h�rbar waren. Endlich
    h�rte das Gefl�ster wieder auf, und Mr. Limbkins begann, nachdem die
    Herren mit feierlicher Miene wieder ihre Pl�tze eingenommen hatten:
    �Wir haben Ihren Vorschlag in Erw�gung gezogen, k�nnen ihn aber nicht
    annehmen.�

    �Unter keinen Umst�nden�, fiel der Herr in der wei�en Weste ein.

    �Ganz entschieden nicht�, erkl�rten die �brigen Mitglieder des
    Kollegiums.

    Da auf Mr. Gamfield der leise Verdacht ruhte, da� schon drei bis vier
    Knaben in seinem Gesch�fte das Leben eingeb��t hatten, so kam ihm der
    Gedanke, das Kollegium k�nnte vielleicht in einer ganz unbegreiflichen
    Laune daran Ansto� genommen haben. Bei der Art ihrer Gesch�ftsf�hrung
    war dies zwar ganz unwahrscheinlich; da er aber keinen besonderen
    Wunsch hegte, diesem Ger�chte neue Nahrung zuzuf�hren, so drehte er
    seine M�tze in den H�nden und entfernte sich langsam von dem Tische.

    �So wollen Sie mir ihn also nicht �berlassen, meine Herren?� fragte
    Gamfield, an der T�re stehenbleibend.

    �Nein�, erwiderte Mr. Limbkins; �wenigstens sind wir der Meinung, Sie
    m��ten mit einer geringeren als der ausgesetzten Summe zufrieden sein,
    da es doch ein gar zu schmutziges Gewerbe ist.�

    Mr. Gamfields Gesicht strahlte, als er rasch an den Tisch zur�ckkehrte
    und sagte: �Was wollen Sie geben, meine Herren? Seien Sie doch nicht zu
    hart gegen einen armen Mann!�

    �Ich sollte meinen, drei Pfund zehn Schilling w�ren �bergenug�, gab Mr.
    Limbkins zur Antwort.

    �Zehn Schilling zu viel�, warf der Herr in der wei�en Weste ein.

    �Nun,� versetzte Gamfield, �sagen wir vier Pfund, meine Herren. Sagen
    wir vier Pfund, und Sie sind ihn auf immer los.�

    �Drei Pfund zehn Schilling�, versetzte Mr. Limbkins fest.

    �Wir wollen den Unterschied teilen, meine Herren, drei Pfund f�nfzehn
    Schilling.�

    �Nicht einen Pfennig mehr�, lautete die feste Entgegnung Mr. Limbkins'.

    �Sie sind verdammt hart gegen mich, meine Herren�, versetzte Gamfield
    niedergeschlagen.

    �Ach, Unsinn�, erwiderte der Herr in der wei�en Weste. �Es ist ein
    gutes Gesch�ft, selbst wenn Sie gar nichts dazu bekommen. Nehmen Sie
    ihn nur, guter Mann. Er ist gerade der richtige Junge f�r Sie. Er
    braucht ab und zu den Stock; das wird ihm sehr gesund sein, und seine
    Bek�stigung braucht auch nicht sehr kostspielig zu werden, denn er ist
    nicht sehr verw�hnt worden, seit er hier geboren wurde. Ha, ha, ha!�

    Mr. Gamfield blickte scheu auf die Herren rund um den Tisch, und
    da er auf den Gesichtern aller ein Schmunzeln bemerkte, l�chelte
    er ebenfalls. Der Handel wurde geschlossen, und Mr. Bumble erhielt
    sofort den Befehl, Oliver Twist am Nachmittag dem Friedensrichter zur
    amtlichen Best�tigung des Lehrvertrages vorzuf�hren.

    Demgem�� wurde der kleine Oliver zu seinem ma�losen Erstaunen aus
    seinem Kerker befreit und erhielt den Befehl, ein frisches Hemd
    anzuziehen. Er hatte kaum diese ungewohnte gymnastische �bung
    beendet, als Mr. Bumble ihm eigenh�ndig einen Napf Hafergr�tze und
    das sonnt�gliche Deputat Brot brachte. Bei diesem furchtbaren Anblick
    begann Oliver bitterlich zu weinen, denn er dachte ganz nat�rlich
    nicht anders, als da� ihn das Kollegium zu irgendeinem n�tzlichen
    Zwecke schlachten lassen wolle, denn sonst h�tte es wohl schwerlich
    angefangen, ihn in dieser Weise fett zu machen.

    �Heul dir die Augen nicht rot, Oliver, sondern i� und sei dankbar�,
    sagte Mr. Bumble in w�rdevollem Tone. �Du sollst in die Lehre gegeben
    werden.�

    �In die Lehre?� fragte das Kind zitternd.

    �Jawohl, Oliver,� erwiderte Mr. Bumble. �Die g�tigen Herren, die ebenso
    viele Eltern f�r dich sind, da du keine eigenen hast, wollen dich in
    die Lehre geben, damit du im Leben auf deinen eigenen F��en stehen
    kannst, und wollen einen Mann aus dir machen, obgleich die Summe,
    die das Kirchspiel daf�r zu bezahlen hat, drei Pfund zehn Schilling
    betr�gt -- drei Pfund zehn Schilling, Oliver! siebzig Schilling --
    einhundertundvierzig Sixpences! und all das f�r ein so ungeratenes
    Waisenkind, das niemand leiden kann.�

    Als Mr. Bumble in seiner Rede innehielt, um Atem zu sch�pfen, rollten
    die Tr�nen dem armen Kinde die Wangen hinunter, und es schluchzte
    bitterlich.

    �Nun, la� gut sein, Oliver�, sagte Mr. Bumble etwas weniger w�rdevoll,
    denn er war mit der Wirkung seiner Beredsamkeit zufrieden. �Wisch
    dir die Augen mit den �rmeln deiner Jacke und weine nicht in deine
    Hafergr�tze. Das ist Dummheit.� Das war es sicherlich, denn es befand
    sich schon gen�gend Wasser darin.

    Auf dem Wege zum Friedensrichter sch�rfte Bumble Oliver auf das
    dringlichste ein, da� alles, was er zu tun h�tte, darin best�nde,
    recht gl�cklich auszusehen, und wenn der alte Herr ihn frage, ob er in
    die Lehre gehen wolle, zu antworten, er freue sich schon sehr darauf.
    Oliver versprach, beiden Weisungen nachzukommen, um so mehr, als Mr.
    Bumble ihm in einem freundlichen Hinweise androhte, es w�rde ihm sonst
    sehr schlecht ergehen. An Ort und Stelle angelangt, wurde er in ein
    kleines Zimmer eingeschlossen, und Mr. Bumble sagte ihm, er solle hier
    bleiben, bis er wiederk�me und ihn abholte.

    So blieb denn der Knabe mit klopfendem Herzen eine halbe Stunde
    allein. Nach deren Verlauf steckte Bumble seinen blo�en, nicht mit dem
    dreieckigen Hut geschm�ckten Kopf herein und sagte laut: �Nun, Oliver,
    mein Kind, komme jetzt zu dem Herrn!�

    W�hrend Mr. Bumble dies sagte, warf er dem Knaben einen grimmigen,
    drohenden Blick zu und f�gte leise hinzu: �Erinnere dich an das, was
    ich dir gesagt habe, infamer Bengel!�

    Oliver starrte bei diesem verschiedenen Ton der Anrede Mr. Bumble
    unschuldig in das Gesicht, aber dieser Herr f�hrte ihn in das
    ansto�ende Zimmer, dessen T�r offen stand, und schnitt ihm dadurch jede
    weitere Bemerkung ab. Es war ein ger�umiges Zimmer mit einem gro�en
    Fenster. Hinter einem Pulte sa�en zwei alte Herren mit gepuderten
    Per�cken, von denen der eine eine Zeitung las, w�hrend der andere
    mit Hilfe einer Schildpattbrille ein kleines vor ihm liegendes St�ck
    Pergament pr�fte. Mr. Limbkins stand vor dem Pulte auf der einen Seite,
    Mr. Gamfield mit teilweise gewaschenem Gesichte auf der anderen.

    Der alte Herr mit der Brille schlief �ber dem St�ck Pergament
    allm�hlich ein, und es entstand eine kurze Pause, nachdem Oliver, von
    Mr. Bumble gef�hrt, sich vor das Pult hingestellt hatte.

    �Dies ist der Knabe, Euer Edeln�, sagte Mr. Bumble.

    Der alte Herr, der die Zeitung las, erhob einen Augenblick den Kopf und
    stie� den anderen alten Herrn an, worauf dieser erwachte.

    �Ah, das ist also der Knabe?� fragte er.

    �Ja, dies ist er, Euer Edeln�, erwiderte Mr. Bumble. �Mache dem Herrn
    Friedensrichter eine Verbeugung, mein Kind.�

    Oliver gehorchte und machte sein sch�nstes Kompliment, das um so tiefer
    ausfiel, da er noch nie Herren mit gepuderten Per�cken gesehen hatte.

    �Der Knabe w�nscht also Schornsteinfeger zu werden?� sagte der
    Friedensrichter.

    �Mit Gewalt,� sagte Bumble, �will's mit Gewalt werden, Euer Edeln;
    w�rde �bermorgen wieder entlaufen, wenn wir ihn morgen in ein anderes
    Gesch�ft g�ben.�

    Der Friedensrichter wendete sich zu dem Schornsteinfeger.

    �Und Sie versprechen, ihn gut zu behandeln, ordentlich zu speisen, zu
    kleiden, und was weiter dazu geh�rt?�

    �Wenn ich's einmal gesagt habe, da� ich's will, so ist's auch meine
    Meinung, da� ich's will�, erwiderte Gamfield barsch.

    �Ihre Rede ist eben nicht fein, mein Freund; doch Sie scheinen ein
    ehrlicher, geradsinniger Mann zu sein�, bemerkte der alte Herr und
    richtete seine Brille auf den Meister, auf dessen h��lichem Gesicht
    die Brutalit�t deutlich zu lesen stand. Aber der Friedensrichter war
    halb blind und halb kindisch, und so konnte man f�glicherweise nicht
    verlangen, da� er das bemerke, was anderen auf den ersten Blick auffiel.

    �Ich hoffe, ich bin es, Sir�, erwiderte Mr. Gamfield grinsend.

    �Ich hege daran nicht den mindesten Zweifel, mein Freund�, erwiderte
    der alte Herr, setzte seine Brille fester auf die Nase und suchte nach
    dem Tintenfa�.

    Es war der kritische Augenblick in Olivers Schicksal. H�tte das
    Tintenfa� dort gestanden, wo es der alte Herr vermutete, so w�rde
    er seine Feder eingetaucht und den Vertrag unterzeichnet haben, und
    Oliver w�re dann auf der Stelle fortgeschleppt worden. Da es sich aber
    unmittelbar vor seiner Nase befand, so folgte daraus mit Notwendigkeit,
    da� er �berall auf dem Pulte nach ihm suchte, ohne es zu finden, und
    da er nun beim Suchen auch gerade vor sich hinblickte, so fiel sein
    Auge auf das bleiche, verst�rte Antlitz Oliver Twists, der trotz aller
    ermahnenden Blicke und P�ffe Bumbles das absto�ende �u�ere seines
    zuk�nftigen Lehrmeisters mit einem aus Grauen und Furcht gemischten
    Ausdruck betrachtete.

    Der alte Herr hielt inne, legte die Feder aus der Hand und blickte von
    Oliver zu Mr. Limbkins hin�ber, der mit unbefangener, heiterer Miene
    eine Prise Schnupftabak zu nehmen versuchte.

    �Mein liebes Kind!� sagte der alte Herr, sich �ber sein Pult lehnend.
    Oliver fuhr beim Klang seiner Stimme zusammen. Dies l��t sich
    entschuldigen, denn die Worte wurden in freundlichem Tone gesprochen,
    und ungewohnte Laute erschrecken jeden. Er zitterte heftig und brach in
    Tr�nen aus.

    �Mein liebes Kind,� begann der alte Herr von neuem, �du siehst bleich
    und ge�ngstet aus. Was ist dir?�

    �Treten Sie ein wenig von ihm weg�, sagte der andere Beamte, das Papier
    weglegend und sich mit einem Ausdrucke reger Teilnahme vorbeugend.

    �Nun, mein Kind, sage uns, was dir ist. Habe keine Furcht!� Oliver fiel
    auf die Knie, hob die gefalteten H�nde empor und flehte schluchzend,
    man m�ge ihn in das finstere Gemach zur�ckbringen, hungern lassen,
    schlagen, ja totschlagen -- nur aber mit dem schrecklichen Manne nicht
    fortschicken.

    �Nun,� sagte Mr. Bumble, indem er seine H�nde mit der eindrucksvollsten
    Feierlichkeit erhob und seine Augen emporschlug, �von allen
    hinterlistigen, niedertr�chtigen Waisenkindern, die ich je gesehen
    habe, bist du der erb�rmlichste Kerl, Oliver.�

    �Halten Sie Ihren Mund, Kirchspieldiener�, rief ihm der zweite alte
    Herr zu, als Mr. Bumble seine Rede beendet hatte.

    �Ich bitte Euer Edeln um Verzeihung�, erwiderte Bumble, der nicht recht
    geh�rt zu haben glaubte. �Haben Euer Edeln zu mir gesprochen?�

    �Jawohl. Halten Sie Ihren Mund!�

    Mr. Bumble war starr vor Entsetzen. Einem Kirchspieldiener zu befehlen,
    den Mund zu halten! Das war ja wirklich eine Umw�lzung aller sittlichen
    Begriffe!

    Der Friedensrichter blickte auf seinen Kollegen, der in bezeichnender
    Weise nickte.

    �Ich mu� dem Vertrage die Best�tigung versagen�, erkl�rte er dann, das
    Pergament unwillig zur Seite schiebend.

    �Ich hoffe,� stotterte Mr. Limbkins, �Sie werden nicht geneigt sein,
    lediglich auf das Zeugnis eines Kindes der Meinung Raum zu geben, da�
    das Verfahren des Direktoriums einem Tadel unterliege.�

    �Ich bin als Friedensrichter nicht berufen, eine Meinung dar�ber
    auszusprechen�, entgegnete der alte Herr. �Nehmen Sie den Knaben wieder
    mit sich und behandeln Sie ihn gut. Er scheint es zu bed�rfen.�

    Man hatte den Anschlag heruntergenommen, am folgenden Morgen wurde
    jedoch Oliver abermals um f�nf Pfund ausgeboten.




    4. Kapitel.

        Oliver Twist, dem eine neue Stellung angeboten wird, tritt in das
        b�rgerliche Leben ein.


    Die Direktoren hatten Bumble befohlen, Erkundigungen einzuziehen, ob
    nicht etwa ein Stromschiffer eines Knaben bed�rfe, wie man denn die
    j�ngeren S�hne und ebenso die Waisenkinder gern zur See schickt, um
    sich ihrer zu entledigen. Gerade als der Kirchspieldiener zur�ckkehrte,
    trat Mr. Sowerberry aus dem Hause, der Leichenbestatter des
    Kirchspiels, der es trotz seiner Besch�ftigung doch nicht wenig liebte,
    zu scherzen.

    �Ich habe soeben das Ma� zu den beiden gestern abend gestorbenen
    Frauenzimmern genommen, Mr. Bumble�, rief er ihm entgegen und bot ihm
    zugleich seine Dose, ein artiges kleines Modell eines Patentsarges.

    �Sie werden noch ein reicher Mann werden, Mr. Sowerberry�, bemerkte
    Bumble.

    �M�cht's w�nschen; aber die Direktoren zahlen nur gar zu geringe
    Preise.�

    �Ihre S�rge sind auch gar zu klein, Mr. Sowerberry.�

    �Gr��ere tun auch nicht not, Mr. Bumble, bei der neuen Speiseordnung.�

    Bumble mi�fiel die Wendung, welche das Gespr�ch genommen; er suchte es
    daher auf einen anderen Gegenstand zu lenken, spielte mit einem seiner
    gro�en Rockkn�pfe mit dem Kirchspielsiegelemblem -- dem barmherzigen
    Samariter -- und begann von Oliver Twist zu sprechen.

    �Beil�ufig,� fing er an, �wissen Sie niemand, der einen Knaben braucht?
    Einen Parochiallehrling, der gegenw�rtig eine blo�e Last, ein dem
    Kirchspiel am Halse h�ngender M�hlstein, m�chte ich sagen, ist. Sehr
    g�nstige Bedingungen, Mr. Sowerberry, sehr g�nstige Bedingungen!�

    Bei diesen Worten erhob Mr. Bumble seinen Stab zu dem Anschlage �ber
    ihm und schlug dreimal auf die Worte �f�nf Pfund�, die mit riesengro�en
    Buchstaben gedruckt waren. Dann fuhr er fort: �Nun, wie denken Sie
    dar�ber?�

    �Oh!� erwiderte der Leichenbestatter; �nun, Sie wissen, Mr. Bumble, ich
    bezahle eine anst�ndige Summe zu den Armenlasten.�

    �Hm!� bemerkte Mr. Bumble. �Nun?�

    �Nun,� antwortete der Leichenbestatter, �ich glaube, da�, wenn ich
    so viel f�r die Armen bezahle, ich auch das Recht habe, so viel wie
    m�glich aus ihnen herauszuschlagen, Mr. Bumble, und so -- und so
    beabsichtige ich denn, den Knaben selber zu nehmen.�

    Mr. Bumble fa�te den Leichenbestatter beim Arme und f�hrte ihn in
    das Haus. Mr. Sowerberry blieb f�nf Minuten bei den Direktoren, und
    es wurde abgemacht, da� Oliver noch am selbigen Abend �auf Probe� zu
    ihm gehen sollte, was soviel sagen will, als da� der Meister, dem ein
    Kirchspielknabe als Lehrling �bergeben wird, denselben auf eine Anzahl
    Lehrjahre haben soll, um mit ihm zu tun, was ihm beliebt, wenn er nach
    kurzer Probezeit ersieht, da� ihm der Knabe genug arbeitet, ohne zu
    e�lustig und also zu kostspielig zu sein. Dem kleinen Oliver wurde
    gesagt, wenn er nicht gutwillig ginge oder sich im Armenhause wieder
    blicken lie�e, so w�rde man ihn nach geb�hrender Z�chtigung zur See
    schicken, wo er unfehlbar ertrinken m�sse. Er zeigte wenig R�hrung und
    wurde nunmehr f�r g�nzlich verh�rtet erkl�rt. Er hatte freilich in
    Wahrheit nicht zu wenig, sondern eher zu viel Gef�hl, war aber durch
    die erfahrene Behandlung bet�ubt und f�r den Augenblick vollkommen
    abgestumpft. Auf dem Wege zu Mr. Sowerberry ermahnte ihn Bumble in
    seinem gew�hnlichen Tone. Oliver traten die Tr�nen in die Augen.

    �Was weinst du, Schlingel? Hab' ich's nicht immer gesagt, da� du die
    schlechteste, undankbarste Kreatur von der Welt bist? Was hast du?
    Sprich!�

    �Ich bin so verlassen, Sir -- so ganz verlassen! Jedermann ist so
    schlimm gegen mich. Es ist mir, als wenn ich hier blutete und mich
    totbluten m��te�; -- und er pre�te die Hand auf das Herz und blickte
    mit nassen Augen seinem F�hrer in das Gesicht.

    Bumble hustete, sagte endlich: �Trockne nur deine Augen und sei ein
    guter Junge�, und ging schweigend weiter.

    Der Leichenbestatter, der soeben die Fensterladen seines Gesch�fts
    geschlossen hatte, machte gerade bei dem Scheine einer elenden Kerze
    einige Eintragungen in sein Rechnungsbuch, als Mr. Bumble eintrat.

    �Aha!� sagte er, von dem Buche aufblickend und mitten in einem Worte
    aufh�rend, �sind Sie es, Bumble?�

    �Niemand anders!� erwiderte der Kirchspieldiener. �Hier ist er! Ich
    habe Ihnen den Knaben gebracht.� Oliver machte eine Verbeugung.

    �Ah, dies ist also der Knabe?� fragte der Leichenbestatter, indem er
    die Kerze in die H�he hob, um Oliver besser betrachten zu k�nnen.
    �Liebe Frau,� rief er dann, �wolltest du vielleicht die Freundlichkeit
    haben, einmal herzukommen?�

    Mrs. Sowerberry tauchte aus einem kleinen Zimmer hinter dem Laden
    auf und zeigte sich in der Gestalt einer kleinen, hageren Frau mit
    z�nkischem Gesichtsausdruck.

    �Liebe Frau,� sagte der Leichenbestatter, �dies ist der Knabe aus dem
    Armenhause, von dem ich dir erz�hlt habe.� Oliver machte abermals eine
    Verbeugung.

    �Mein Himmel, wie klein er ist!� rief Mrs. Sowerberry aus.

    �Er ist allerdings klein�, sagte Bumble, Oliver sehr unwillig
    anblickend, als ob es des Knaben Schuld gewesen w�re, da� er nicht
    gr��er war; �er wird aber gr��er werden, Mrs. Sowerberry.�

    �O ja, auf unsere Kosten�, entgegnete sie verdrie�lich. �Ich sehe keine
    Ersparnis mit Kirchspielkindern; sie kosten allezeit mehr, als sie wert
    sind. Die M�nner glauben aber immer, alles am besten zu wissen.�

    Bei diesen Worten �ffnete sie eine Seitent�r und stie� Oliver eine
    Treppe hinunter in ein finsteres, dumpfes Gela�, den Vorraum des
    Kohlenkellers und �K�che� genannt, und befahl einer schlumpigen
    Dienstmagd, ihm zu geben, was f�r den nicht nach Hause gekommenen Trip
    zur�ckgestellt w�re.

    O da� doch so mancher, dessen Blut von Eis und dessen Herz von Stein
    ist und der dennoch eine Stimme sich anma�t, eine Stimme hat, wo es der
    Beurteilung der Lage, dem Wohl oder Wehe der Armen gilt, den Knaben
    h�tte verschlingen sehen k�nnen, was der Haushund verschm�ht! Wie sehr
    w�re so vielen Menschenfreunden dieselbe und keine andere Di�t zu
    w�nschen!

    Frau Sowerberry hatte dem Knaben in stummem Entsetzen und mit tr�ben
    Ahnungen in betreff seines k�nftigen Appetits zugeschaut; er h�rte auf
    zu essen, als er nichts mehr fand.

    �Bist du endlich fertig?� sagte sie. �Nun komm, dein Bett ist unter
    dem Ladentische. Du wirst dich doch nicht grauen, zwischen S�rgen zu
    schlafen? Aber wenn du auch nicht wolltest, du bekommst keine andere
    Schlafstelle.�

    Oliver folgte sch�chtern und geduldig seiner neuen Herrin.




    5. Kapitel.

        Oliver unter neuen Umgebungen und bei einem Leichenbeg�ngnisse.


    Sobald Oliver im Laden des Leichenbestatters allein gelassen
    war, setzte er seine Lampe auf eine Bank, und Furcht und Grauen
    durchschauerte ihn. Mitten im Gemach stand ein neuer, fast fertiger
    Sarg; die schon zugeschnittenen, an die W�nde umher gelehnten Bretter
    erschienen ihm beim matten Lampenlichte wie Geister. Auf dem Boden
    lagen gro�e N�gel, Holzsp�ne, St�cke schwarzen Tuchs und Sargembleme,
    und an der Wand �ber dem Ladentische hing das grauenhafte Bild eines
    Leichenzuges. Die Luft war dr�ckend hei�; sie deuchte Oliver wie
    Grabesluft, die �ffnung zu seiner Ruhest�tte unter dem Ladentische wie
    ein g�hnendes Grab.

    Er f�hlte sich allein und verlassen in der Welt, und obwohl er keinen
    Schmerz �ber Trennung von Freunden oder Angeh�rigen empfand, so war ihm
    das Herz dennoch schwer; und als er in sein enges Bett hineinkroch,
    w�nschte er, da� es sein Sarg sein und da� er darin hinaus auf den
    Kirchhof getragen werden m�chte, wo das hohe stille Gras �ber ihm
    w�chse und im Winde s�uselte und das L�uten der alten, traurigen
    Turmglocke ihm sch�ne Tr�ume zuf�hrte in seinem s��en Schlummer.

    Er wurde am folgenden Morgen durch ein lautes Pochen an der Ladent�r
    aus seinem unruhigen Schlafe geweckt; dasselbe wiederholte sich, ehe er
    in seine Kleider schl�pfen konnte, ungef�hr f�nfundzwanzigmal und in
    ungest�mer Weise. Als er die Kette zu l�sen begann, h�rten die Beine zu
    sto�en auf, und eine Stimme lie� sich vernehmen.

    ��ffne die T�r, wird's bald?� rief die Stimme, die zu den Beinen
    geh�rte.

    �Sofort, Sir!� erwiderte Oliver, indem er die Kette losmachte und den
    Schl�ssel umdrehte.

    �Ich vermute, du bist der neue Lehrjunge, nicht wahr?� sprach die
    Stimme durch das Schl�sselloch.

    �Ja, Sir!� antwortete Oliver.

    �Wie alt bist du?� fragte die Stimme weiter.

    �Zehn Jahre, Sir!� entgegnete Oliver.

    �Dann werde ich dich pr�geln, wenn ich hineinkomme�, sagte die Stimme;
    �du wirst gleich sehen, da� ich es tue, du Armenh�usler!�

    Oliver hatte schon zu oft das angedrohte Schicksal �ber sich ergehen
    lassen m�ssen, um den leisesten Zweifel zu hegen, da� der Besitzer der
    Stimme, wer es auch sein mochte, sein Versprechen wahr machen w�rde. Er
    schob den Riegel mit zitternder Hand zur�ck und �ffnete die T�r.

    Ein paar Sekunden lang blickte Oliver die Stra�e auf und ab, weil er
    glaubte, der unbekannte Besucher, der ihn durch das Schl�sselloch
    angeredet hatte, habe sich einige Schritte entfernt, um sich zu
    erw�rmen; denn es war niemand zu sehen, au�er einem gro�en Armenknaben,
    der auf einem Pfosten vor dem Hause sa� und ein Butterbrot verzehrte.

    �Verzeihen Sie, Sir,� sagte Oliver endlich, da er keinen anderen
    Besucher erblicken konnte, �haben Sie geklopft?�

    �Ja, ich habe mit den F��en an die T�r gesto�en�, erwiderte der
    Armenknabe.

    �W�nschen Sie einen Sarg, Sir?� fragte Oliver unschuldig.

    �Es wird nicht lange w�hren, bis du selbst einen brauchst,� war die
    zornige Antwort, �wenn du Scherz mit Leuten treibst, die dir zu
    befehlen haben. Wei�t du nicht, wer ich bin? Noah Claypole, und du bist
    mir untergeben, Musj� Ohnevater. �ffne die Fensterl�den, Faulpelz!�

    Oliver tat, wie ihm gehei�en war, und gleich darauf erschien Mr. und
    Mrs. Sowerberry. Oliver und sein neuer Tyrann wurden in die K�che
    geschickt, um ihr Fr�hst�ck zu erhalten. Charlotte, die K�chin,
    bedachte Noah gut und Oliver desto schlechter, der obendrein von jenem
    sehr unsanft in einen dunklen Winkel gesto�en und vielfach geh�nselt
    wurde.

    Noah war ein Freisch�ler, aber doch keine Waise aus dem Armenhause.
    Sein Stammbaum war ihm sehr wohl bekannt; seine Eltern wohnten in der
    Nachbarschaft. Seine Mutter war eine Waschfrau und sein Vater ein
    pensionierter, t�glich betrunkener Soldat. Die Ladenburschen nannten
    ihn ver�chtlich �Lederhose� und so fort, was er schweigend duldete,
    dagegen aber nunmehr mit desto gr��erem �bermut einen Schw�cheren und
    Elternlosen behandelte, den er als solchen tief unter sich sah. --
    Welch ein k�stlicher Stoff zu Betrachtungen �ber die liebensw�rdige
    menschliche Natur, deren vortreffliche Eigenschaften sich beim
    hochstehenden Lord wie beim Armenknaben offenbaren!

    Oliver hatte sich drei bis vier Wochen bei Mr. Sowerberry befunden,
    als derselbe einst gegen seine Hausehre die Rede auf ihn brachte. �Der
    Knabe sieht wirklich gut aus�, bemerkte er.

    �Kein Wunder,� entgegnete sie, �denn er i�t genug.�

    �Er hat ein �u�erst melancholisches Gesicht und sieht immer so
    tr�bselig aus, da� er wirklich einen vortrefflichen Stummen[A] abgeben
    w�rde.�

      [A] Die stummen Diener des Leichenbestatters, die vor den T�ren der
      Trauerh�user stehen.

    Seine Gattin sah ihn verwundert an, und er fuhr fort: �Ich meine nicht
    bei Erwachsenen, sondern bei Kinderbegr�bnissen. 's ist etwas Neues,
    auch zu dergleichen kleine Stumme zu stellen, und man kann sich etwas
    davon versprechen.�

    Mrs. Sowerberry, die f�r Gesch�ftssachen ein gutes Verst�ndnis besa�,
    war von der Neuheit des Gedankens �berrascht; da es aber gegen ihre
    W�rde versto�en haben w�rde, wenn sie dies zugegeben h�tte, so fragte
    sie nur mit gro�er Sch�rfe im Ton, warum ihr einf�ltiger Eheherr denn
    nicht schon l�ngst daran gedacht habe, und Mr. Sowerberry, der dies
    richtig als Zustimmung auslegte, beschlo�, Oliver in die Mysterien
    des Leichenbestattergesch�ftes einzuweihen und sich daher von ihm
    zum ersten besten vorkommenden Begr�bnisse begleiten zu lassen. Die
    Gelegenheit lie� nicht lange auf sich warten, denn eine halbe Stunde
    darauf erschien Bumble mit dem Auftrage zu einem Kirchspielbegr�bnisse.

    Mr. Sowerberry ordnete die erforderlichen Vorbereitungen an und befahl
    Oliver, mit ihm zu gehen. Sie begaben sich nach dem bezeichneten Hause,
    um das Ma� zum Sarge zu nehmen, wo sich ihren Blicken eine Szene des
    grauenvollsten Elends darbot, die auf Oliver, obgleich er an Elend so
    wohl gew�hnt war, den peinlichsten Eindruck machte.

    Am folgenden Tage, der rauh und regnerisch war, wiederholten sie
    ihren Besuch, die Leiche wurde in den Sarg gelegt, jede Anordnung war
    getroffen. Mr. Sowerberry sagte den Tr�gern, sie m�chten sich sputen
    und den Geistlichen nicht warten lassen; es w�re schon sp�t. Die Tr�ger
    setzten sich in eine Art von Trab, und Oliver mu�te fast laufen, um
    mitkommen zu k�nnen. Der Geistliche war noch nicht angelangt, der
    Sarg wurde in einem entfernten Winkel des Kirchhofs neben der Gruft
    einstweilen niedergesetzt, und Mr. Sowerberry und Bumble setzten sich
    zum K�ster in die Sakristei an das Feuer und nahmen die Zeitungen zur
    Hand.

    Nach einer halben Stunde erschien der Geistliche, Bumble verjagte die
    Gassenbuben, die sich damit unterhielten, her- und hin�ber �ber den
    Sarg zu springen, der Geistliche las eilend die Gebete, entfernte sich
    wieder, der Sarg wurde eingesenkt, die Grube zugeworfen, und alle
    begaben sich auf den Heimweg.

    �Nun, Oliver, wie hat dir's gefallen?� fragte Mr. Sowerberry.

    �Recht gut, bedanke mich, Sir!� antwortete Oliver z�gernd. �Aber doch
    eigentlich nicht sehr gut.�

    �Wirst dich schon daran gew�hnen�, sagte der Leichenbesorger; �und 's
    ist gar nichts, wenn du's erst gewohnt bist.�

    Oliver h�tte gern gewu�t, wie lange es gedauert, ehe Mr. Sowerberry
    sich daran gew�hnt, wagte jedoch nicht zu fragen und kehrte
    gedankenvoll mit seinem Herrn nach Hause zur�ck.




    6. Kapitel.

        In welchem Oliver kr�ftig auftritt.


    Es trat gerade eine sehr ungesunde Zeit ein, und Oliver sammelte daher
    in wenigen Wochen viel Erfahrung. Die Erfolge der scharfsinnigen
    Spekulation Mr. Sowerberrys �bertrafen alle seine Erwartungen. Die
    �ltesten Leute wu�ten sich nicht zu erinnern, da� so viele Kinder an
    den Masern gestorben waren, und Oliver mit schwarzen, bis an die Knie
    herunterreichenden Hutb�ndern f�hrte einen Leichenzug nach dem andern
    an. Die M�tter bewunderten ihn �ber die Ma�en und waren unbeschreiblich
    ger�hrt. Da er seinen Herrn auch zu den meisten Begr�bnissen von
    Erwachsenen begleiten mu�te, um sich die f�r einen vollkommenen
    Leichenbestatter so notwendige gemessene Ruhe und Selbstbeherrschung
    anzueignen, so hatte er h�ufig Gelegenheit, die sch�ne Ergebung und
    Seelenst�rke zu bemerken, welche so viele Leute bei ihren schmerzlichen
    Pr�fungen und Verlusten beweisen.

    Hatte Sowerberry zum Beispiel das Begr�bnis einer reichen alten Dame
    oder eines reichen alten Herrn zu besorgen, der von einer gro�en
    Anzahl von Neffen und Nichten umgeben war, welche sich w�hrend seiner
    Krankheit vollkommen untr�stlich gezeigt und ihren Schmerz nicht einmal
    vor den Augen des gro�en und gr��ten Publikums hatten bemeistern
    k�nnen, so blieb es selten aus, da� sie unter sich so heiter waren,
    als man es nur w�nschen konnte, und so froh und zufrieden miteinander
    redeten oder auch lachten, als wenn sie ganz und gar keine Tr�bsal
    erlebt h�tten. Ehem�nner ertrugen den Verlust ihrer Frauen mit der
    heldenm�tigsten Ruhe, und Ehefrauen legten die Trauerkleider um ihre
    M�nner auf eine Weise an, als wenn sie dadurch nicht etwa Schmerz
    andeuten, sondern so anziehend als m�glich erscheinen wollten. Viele
    Damen und Herren, welche bei der Beerdigung der Verzweiflung nahe
    zu sein schienen, beruhigten sich schon auf dem Heimwege und waren
    vollkommen gefa�t, bevor die Teestunde vor�ber war. Dieses alles war
    sehr angenehm und lehrreich anzuschauen, und Oliver sah es mit gro�er
    Bewunderung.

    Da� das Beispiel so vieler Leidtragenden ihn zur Ergebung und Geduld
    gestimmt h�tte, kann ich mit Bestimmtheit nicht behaupten, sondern
    vermag nur so viel zu sagen, da� er wochenlang mit Sanftmut die
    Tyrannei und �ble Behandlung ertrug, die er von seiten Noahs erfuhr,
    der um so erbitterter gegen ihn wurde, weil sein Neid gegen ihn
    erregt worden war. Charlotte mi�handelte ihn, weil es Noah tat, und
    Mrs. Sowerberry war seine erkl�rte Feindin, weil ihr Gatte sich ihm
    ziemlich freundlich erwies. Und so befand sich denn Oliver bei diesen
    Feindschaften und fortw�hrender Leichenbegleitungslast nicht ganz so
    behaglich wie das hungrige Ferklein, das aus Versehen in die Kornkammer
    einer Brauerei eingeschlossen war.

    Es mu� aber jetzt ein an sich unbedeutender Vorfall erz�hlt werden, der
    jedoch eine bedeutende Ver�nderung mit Oliver selbst wie mit seinen
    Lebensschicksalen zur Folge hatte.

    Sein Peiniger trieb seine gew�hnlichen Neckereien weiter als gew�hnlich
    und hatte es offenbar darauf angelegt, ihn au�er Fassung und zum Weinen
    zu bringen, was ihm jedoch nicht gelingen wollte. Endlich sagte Noah
    scherzend, er werde nicht verfehlen zuzuschauen, wenn Oliver geh�ngt
    w�rde, und f�gte hinzu: �Was wird aber deine Mutter dazu sagen -- und
    wie geht's ihr denn?�

    �Sie ist tot�, entgegnete Oliver; �untersteh dich aber nicht, mir etwas
    Schlechtes �ber sie zu sagen.�

    Oliver wurde feuerrot, als er das sagte; er atmete rasch, um Mund
    und Nase zuckte es ihm eigent�mlich, und Claypole hielt dies f�r ein
    untr�gliches Anzeichen, da� Oliver bald heftig weinen werde. In dieser
    �berzeugung ging er in seiner Qu�lerei weiter.

    �Woran starb sie denn, Armenh�usler?� fragte er.

    �An Kummer und Herzleid, wie mir eine unserer alten W�rterinnen gesagt
    hat,� erwiderte Oliver, mehr, wie wenn er mit sich selbst redete, als
    Noahs Frage beantwortend. �Ich glaube, da� ich's wei�, was es hei�t,
    daran zu sterben!�

    �ber seine Wange rollte eine Tr�ne hinab, Noah pfiff eine muntere Weise
    und sagte darauf: �Was hast du denn zu pl�rren -- um deine Mutter?�

    �Da� du mir kein Wort mehr von ihr sagst -- sonst nimm dich in acht!�
    rief Oliver.

    �Ich soll mich in acht nehmen -- ich -- mich in acht nehmen vor einem
    solchen unversch�mten Tunichtgut? Und von wem soll ich kein Wort mehr
    sagen? Von deiner Mutter? Die mag auch die rechte gewesen sein -- ha,
    ha, ha!�

    Oliver verbi� seine Pein und schwieg. Noah nahm den Ton sp�ttischen
    Mitleids an.

    �Nun, nun, sei nur ruhig; 's ist nichts mehr dran zu �ndern, und ich
    bedaure dich, wie's alle tun. Indes ist das wahr, ich wei� es, deine
    Mutter taugte nichts; sie ist eine ganz verworfene Person gewesen.�

    �Was sagst du?� rief Oliver rasch aufblickend.

    �Eine ganz verworfene Person,� erwiderte Noah k�hl, �und es war nur
    gut, da� sie starb, denn es w�rde ihr jetzt schlecht genug ergehen in
    der Tretm�hle, wenn sie anders nicht deportiert oder geh�ngt worden
    w�re. Hab' ich nicht recht, Armenh�usler?�

    Olivers Geduld war zu Ende; purpurrot vor Wut sprang er auf, warf
    seinen Stuhl samt dem Tische um, fa�te Noah bei der Kehle, sch�ttelte
    ihn so stark, da� ihm die Z�hne im Munde klapperten, sammelte seine
    ganze Kraft und schlug ihn mit einem einzigen Schlage zu Boden.

    Eine Minute vorher hatte er das Aussehen des stillen, sanftm�tigen,
    eingesch�chterten Kindes noch gehabt, zu dem harte Behandlung ihn
    gemacht hatte. Aber sein Mut war endlich erwacht; die t�dliche
    Beleidigung, die Noah seiner toten Mutter zugef�gt, hatte sein Blut
    in Wallung gebracht. Seine Brust hob sich, er stand aufrecht da wie
    ein Held, sein Auge strahlte lebhaft; sein ganzes Wesen war ver�ndert,
    als er funkelnden Blickes vor dem feigen Qu�ler stand, der jetzt
    zusammengekr�mmt zu seinen F��en lag.

    �Er ermordet mich!� heulte Noah. �Charlotte, Fr�ulein! Der neue
    Lehrjunge ermordet mich! Zu Hilfe, zu Hilfe! Oliver ist verr�ckt
    geworden! Char--lotte!�

    Noahs Geschrei wurde durch ein lautes Aufkreischen von Charlottes Seite
    und durch ein lauteres von seiten Mrs. Sowerberrys beantwortet; die
    erstere st�rzte durch eine Seitent�r in die K�che, w�hrend die letztere
    noch auf der Treppe zauderte, bis sie sich v�llig davon �berzeugt
    hatte, da� sie n�her treten konnte, ohne ihr kostbares Leben zu
    gef�hrden.

    �Du verdammter Halunke!� schrie Charlotte und packte Oliver kr�ftig am
    Arme. �Du undankbarer, mordgieriger, abscheulicher Schuft!� Und dabei
    schlug sie unausgesetzt aus Leibeskr�ften auf Oliver ein.

    Charlottes Faust geh�rte nicht zu den leichtesten, und jetzt kam ihr
    auch noch Mrs. Sowerberry zu Hilfe, die in die K�che st�rzte und ihn
    mit der einen Hand festhielt, w�hrend sie ihm mit der anderen das
    Gesicht zerkratzte. Bei diesem g�nstigen Stande der Angelegenheit erhob
    sich auch Noah vom Fu�boden und griff ihn von hinten an.

    Dieser dreifache Angriff war zu heftig, als da� er lange h�tte dauern
    k�nnen. Als sie alle drei erm�det waren und nicht l�nger zerren und
    schlagen konnten, schleppten sie Oliver in den Kehrichtkeller und
    schlossen ihn hier ein. Nachdem dies gl�cklich vollbracht war, sank
    Mrs. Sowerberry auf einen Stuhl und brach in Tr�nen aus.

    �Um Gottes willen, sie stirbt!� rief Charlotte. �Ein Glas Wasser,
    liebster Noah! Spute dich!�

    �O Charlotte�, sagte Mrs. Sowerberry st�hnend, �was f�r ein Gl�ck, da�
    wir nicht alle in unseren Betten ermordet worden sind!�

    �Ja, Madam,� lautete die Antwort, �das ist in der Tat ein Gl�ck von
    Gott. Der arme Noah! Er war schon halb ermordet, als ich hineinkam.�

    �Armer Junge!� sagte Mrs. Sowerberry, indem sie mitleidig auf den
    Knaben blickte. �Was sollen wir anfangen?� fuhr sie nach einer Weile
    fort. �Der Herr ist nicht daheim; es ist kein Mann im ganzen Hause, und
    er wird die Kellert�r in zehn Minuten eingesto�en haben.�

    �Mein Gott, mein Gott!� jammerte Charlotte, �ich wei� es nicht, Ma'am!
    Aber vielleicht schicken wir nach der Polizei.�

    �Oder nach dem Milit�r!� warf Claypole ein.

    �Nein, nein!� erwiderte Mrs. Sowerberry, die sich in diesem Augenblick
    an Olivers alten Freund erinnerte. �Lauf zu Mr. Bumble, Noah, und bitte
    ihn, unverz�glich herzukommen und keine Minute zu verlieren. Es tut
    nichts, wenn du auch ohne M�tze gehst. Mach hurtig!�

    Ohne sich die Zeit zu einer Antwort zu lassen, st�rzte Noah davon,
    und die ihm begegnenden Leute waren sehr erstaunt, einen Armenknaben
    barh�uptig in voller Eile durch die Stra�en rennen zu sehen.




    7. Kapitel.

        Oliver bleibt widerspenstig.


    Noah Claypole unterbrach seinen hastigen Lauf nicht ein einziges Mal
    und kam ganz atemlos vor dem Tor des Armenhauses an. Hier blieb er
    einen Augenblick stehen, um sein Gesicht in m�glichst kl�gliche Falten
    zu legen, klopfte dann laut an die Pforte und zeigte dem �ffnenden
    Armenh�usling eine so jammervolle Miene, da� selbst dieser, der sein
    ganzes Leben lang nichts als jammervolle Mienen um sich gesehen hatte,
    erschrocken zur�ckfuhr und fragte: �Was hast du denn nur, Junge?�

    �Mr. Bumble, Mr. Bumble!� rief Noah in gut geheuchelter Angst und in so
    lautem, erregtem Tone, da� Mr. Bumble, der zuf�llig in der N�he war,
    es nicht nur h�rte, sondern auch dadurch in solche Aufregung geriet,
    da� er ohne seinen dreieckigen Hut in den Hof st�rzte -- ein deutlicher
    Beweis daf�r, da� selbst ein Kirchspieldiener unter Umst�nden seine
    Fassung verlieren und seine pers�nliche W�rde au�er acht lassen kann.

    �Oh, Mr. Bumble -- o Sir!� schrie Noah; �Oliver, Sir -- Oliver Twist!�

    �Wie -- was? Ist er -- ist er davongelaufen?�

    �Nein, Sir; er ist ganz ruchlos geworden. Er hat mich und Charlotte und
    Missis ermorden wollen! O Sir! o Sir -- mein Nacken, mein Kopf, mein
    Leib, mein Leib!�

    Sein Geheul zog den Herrn mit der wei�en Weste herbei.

    �Sir,� rief Bumble demselben entgegen, �hier ist ein Knabe aus der
    Freischule, der von Oliver Twist beinahe ermordet worden w�re!�

    �Bei Gott,� bemerkte der Herr mit der wei�en Weste, �das habe ich
    gewu�t. Ich hatte von Anfang an eine seltsame Ahnung, da� dieser
    freche, kleine Taugenichts noch geh�ngt werden w�rde.�

    �Er hat auch die Magd ermorden wollen�, sagte Bumble mit bleichem
    Gesicht.

    �Und die Frau�, fiel Noah ein.

    �Und nicht wahr, Noah, sagtest du nicht, auch seinen Herrn?� fragte
    Bumble.

    �Nein, der Herr war nicht zu Hause, sonst h�tte er ihn auch gemordet�,
    antwortete Noah. �Aber der B�sewicht sagte, er wollte es tun.�

    �Sagte er, da� er es tun wollte, mein Kind?� fragte der Herr mit der
    wei�en Weste.

    �Ja, Sir!� erwiderte Noah. �Und Missis w�nscht zu wissen, ob Mr.
    Bumble wohl nicht einen Augenblick Zeit h�tte, um zu kommen und ihn zu
    z�chtigen, da der Herr nicht zu Hause ist.�

    �Gewi�, mein Junge, gewi߻, sagte der Herr in der wei�en Weste, indem
    er freundlich l�chelte und Noahs Kopf streichelte. �Du bist ein guter
    Junge, ein sehr guter Junge. Hier hast du einen Penny. Bumble, gehen
    Sie sofort mit Ihrem Stabe zu Sowerberry und sehen Sie zu, was am
    besten zu tun ist. Schonen Sie ihn nicht, Bumble, und sagen Sie auch
    Sowerberry, er solle in Zukunft strenge mit ihm verfahren.�

    �Ich werde alles zu Ihrer vollen Zufriedenheit besorgen, Sir!�
    erwiderte Bumble, indem er sich zusammen mit Noah auf den Weg machte.

    Als sie an ihrem Bestimmungsorte anlangten, war die Lage der Dinge
    dort unver�ndert. Sowerberry war noch nicht zur�ckgekehrt, und Oliver
    schlug fortw�hrend mit unverminderter Heftigkeit an die Kellert�r. Mr.
    Bumble donnerte mit seinem Fu�e von au�en an die T�r, um sein Kommen
    anzuzeigen, legte dann seinen Mund ans Schl�sselloch und sagte in
    tiefem, eindringlichem Tone: �Oliver.�

    �La�t mich hinaus!� rief Oliver von innen.

    �Kennst du meine Stimme, Oliver?�

    �Ja!�

    �F�rchtest du dich nicht -- zitterst du nicht bei meiner N�he?�

    �Nein!�

    Bumble war starr vor Erstaunen.

    �Er mu� verr�ckt geworden sein!� bemerkte Mrs. Sowerberry.

    �'s ist keine Verr�cktheit, Ma'am,� sagte Bumble, �'s ist das Fleisch!�

    �Das Fleisch?!�

    �Ja, ja, Ma'am! Sie haben ihn �berf�ttert, Ma'am. H�tten Sie ihm nichts
    als Haferbrei gegeben, so w�rde er nimmermehr so geworden sein.�

    Mrs. Sowerberry machte sich wegen ihrer Gutherzigkeit und Freigebigkeit
    die bittersten Vorw�rfe, so unschuldig in Gedanken, Worten und Werken
    sie auch war.

    Bumble erkl�rte, da� nur Einsperren und sodann strenge Di�t den
    rebellischen Sinn des kleinen Galgenstricks w�rden b�ndigen k�nnen.
    In diesem Augenblick kehrte Sowerberry zur�ck, dem sofort der Vorfall
    mit solchen �bertreibungen erz�hlt wurde, da� er die T�r �ffnete, den
    Knaben beim Kragen fa�te und herauszog.

    Olivers Kleider waren zerrissen, sein Gesicht war verschwollen und
    zerkratzt, und sein Haar hing ihm wirr �ber die Stirn herab. Die
    zornige R�te war jedoch aus seinem Gesicht nicht verschwunden, und als
    er aus seinem Gef�ngnis gezogen wurde, warf er Noah einen drohenden
    Blick zu.

    �Nun, du bist ja ein netter Bursche�, sagte Sowerberry, sch�ttelte
    Oliver derb und gab ihm rechts und links ein paar Ohrfeigen.

    �Er beschimpfte meine Mutter�, sagte Oliver.

    �Und wenn er das auch tat, du undankbarer B�sewicht�, versetzte Mrs.
    Sowerberry. �Sie hat's verdient, was er von ihr gesagt hat, und noch
    viel mehr.�

    �Nein, nein!� rief Oliver. �'s ist eine L�ge!�

    Mrs. Sowerberry brach in eine Tr�nenflut aus, und dies lie� ihrem
    Gatten keine Wahl. Denn wenn er nicht auf der Stelle Oliver
    nachdr�cklich gez�chtigt h�tte, so w�rde er sich, gem�� allen
    Ehez�nkereiregeln, als eine Nachtm�tze, ein liebloser Ehemann, ein
    Ungeheuer gezeigt haben. So ungern er es daher auch tun mochte, er
    z�chtigte Oliver derma�en, da� die nachtr�gliche Anwendung des Rohrs
    Mr. Bumbles jedenfalls sehr unn�tig war. Oliver wurde darauf bei Wasser
    und Brot wieder eingesperrt und sp�t abends unter Noahs unbarmherzigem
    Gesp�tt zu Bett gewiesen.

    Erst hier lie� er seinen Gef�hlen freien Lauf. Er hatte allen Spott und
    Hohn mit hartn�ckiger Verachtung, die schmerzlichsten Streiche ohne
    Schrei ertragen und w�rde nicht geweint haben, wenn man ihn lebendig
    ger�stet h�tte; ein solcher Stolz war in seiner Brust erwacht. Nun
    aber, da er allein und g�nzlich sich selber �berlassen war, fiel er auf
    die Knie nieder, bedeckte das Gesicht mit den H�nden und weinte solche
    Tr�nen, wie Gott sie den Betr�bten und Ge�ngsteten zur Erleichterung
    ihres Herzens sendet, wie nur wenige menschliche Wesen, so jung an
    Jahren wie Oliver, sie zu vergie�en Ursache hatten.

    Es w�hrte lange, bevor er sich wieder erhob. Das Licht war tief
    heruntergebrannt, er horchte und blickte vorsichtig umher, �ffnete
    leise die T�r und sah hinaus. Die Nacht war finster und kalt. Die
    Sterne schienen ihm weiter von der Erde entfernt zu sein, als er sie je
    gesehen; die B�ume, von keinem Winde bewegt, standen wie Geister da. Er
    verschlo� die T�r wieder, kn�pfte seine wenigen Habseligkeiten in ein
    Taschentuch und setzte sich auf eine Bank, um den Anbruch des Tages zu
    erwarten.

    Mit dem ersten durch die Ritzen der Fensterladen eindringenden
    Lichtstrahle stand er auf, �ffnete die T�r zum zweiten Male, blickte
    furchtsam umher, z�gerte ein paar Augenblicke, trat hinaus und ging,
    ungewi�, wohin er sich wenden sollte, rasch vorw�rts. Nach einiger
    Zeit gewahrte er, da� er sich ganz in der N�he der Anstalt bef�nde, in
    der er seine ersten Kinderjahre verlebt hatte. Es war niemand zu h�ren
    oder zu sehen; er blickte in den Garten hinein. Einer seiner kleinen,
    weit j�ngeren Spielkameraden reinigte ein Beet vom Unkraut. Sie hatten
    miteinander gar oft Hunger, Schl�ge und Einsperrung erduldet.

    �Pst! Dick!� rief Oliver.

    Der Knabe lief herbei und streckte ihm die abgemagerten H�nde durch die
    Gittert�r entgegen.

    �Ist schon jemand auf, Dick?�

    �Keiner als ich.�

    �Sag' ja nicht, da� du mich gesehen hast, Dick; ich bin fortgelaufen;
    konnt's nicht mehr aushalten und will mein Gl�ck in der Welt versuchen.
    Ich mu� weit fort von hier; wei� nicht, wohin. Wie bla� du aussiehst!�

    �Ich habe den Doktor sagen h�ren, da� ich sterben m��te. Ach, das ist
    sch�n, da� du hier bist! Aber halt dich nicht auf; lauf fort!�

    �Ja, ja, leb wohl! Ich wei� gewi�, wir sehen uns wieder, Dick. Du wirst
    noch recht gl�cklich werden.�

    �Das hoff' ich -- wenn ich tot bin; eher nicht. Ich wei� es, Oliver,
    der Doktor hat recht; denn ich tr�ume so viel vom Himmel und von Engeln
    und freundlichen Gesichtern, die ich niemals sehe, wenn ich aufwache.
    Leb wohl, Oliver; geh mit Gott! Gottes Segen begleite dich!�

    Oliver hatte noch nie des Himmels Segen auf sich herabrufen h�ren, und
    nie verga� er diese Segnung von den Lippen eines Kindes unter allen
    Leiden, Sorgen, M�hen, K�mpfen und Wechselschicksalen seines Lebens.




    8. Kapitel.

        Oliver geht nach London und trifft mit einem absonderlichen jungen
        Gentleman zusammen.


    Oliver lief ohne Rast und Ruhe, bis er um die Mittagsstunde bei einem
    Meilensteine stillstand, auf dem die Entfernung Londons angegeben
    war. Dort konnte man ihn nicht finden, er hatte oft sagen h�ren, da�
    die unerme�liche Stadt zahllose Mittel zum Fortkommen darb�te, sein
    Entschlu� war gefa�t; er machte sich bald wieder auf den Weg und
    gedachte nun erst der Schwierigkeiten, die er zu �berwinden haben
    w�rde, um an sein Ziel zu gelangen. Er hatte ein grobes Hemd, zwei
    Paar Str�mpfe, eine Brotrinde und einen Penny in seinem B�ndel -- ein
    Geschenk Mr. Sowerberrys nach einem Begr�bnisse, bei welchem er sich
    dessen ungew�hnliche Zufriedenheit verdient hatte. Er sann vergeblich
    dar�ber nach, wie er mit so geringen Mitteln London erreichen solle --
    und trabte weiter.

    Nachdem er zwanzig Meilen zur�ckgelegt hatte, lenkte er auf eine Wiese
    ein und legte sich in einem Heuhaufen zur Ruhe nieder. Er machte am
    zweiten Tage abermals zw�lf Meilen, verwendete seinen Penny f�r Brot,
    �bernachtete auf �hnliche Weise und erhob sich am dritten Morgen fast
    erfroren und mit erstarrten Gliedern, so da� er sich kaum von der
    Stelle bewegen konnte.

    Die Stra�e wand sich hier einen ziemlich steilen H�gel hinauf, und er
    flehte die Au�enpassagiere einer Postkutsche um eine Gabe an. Nur einer
    beachtete ihn, rief ihm zu, er m�ge warten, bis man oben angelangt
    w�re, und begehrte darauf zu erfahren, wie weit er um einen halben
    Penny mitlaufen k�nne. Oliver mu�te nach der gr��ten Anstrengung doch
    bald zur�ckbleiben, und der Mildt�tige steckte sein Geldst�ck wieder
    in die Tasche und erkl�rte ihn f�r einen faulen Schlingel, der keine
    Freigebigkeit verdiene. Dahin rollte die Postkutsche und lie� nur eine
    Staubwolke zur�ck.

    In manchen D�rfern waren Pfosten mit Tafeln errichtet, auf welchen
    scharfe Drohungen gegen alle Bettler zu lesen waren, und Oliver eilte
    furchtsam weiter; in anderen, wenn er etwa vor einem Gasthause mit
    sehns�chtigen Blicken stillstand, hie� man ihn sich davonmachen, wenn
    er nicht als ein Dieb eingesperrt werden wollte. Aus vielen H�usern
    vertrieb ihn die Drohung, da� man die Hunde loslassen werde, wenn er
    sich nicht sofort entferne.

    Es w�rde ihm ohne Zweifel ergangen sein, wie seiner ungl�cklichen
    Mutter, wenn sich nicht ein menschenfreundlicher Schlagbaumw�rter und
    eine gutherzige Frau seiner angenommen h�tten. Jener erquickte ihn
    durch ein, wenn auch nur aus Brot und K�se bestehendes Mittagsmahl;
    und diese, die einen schiffbr�chigen, sie wu�te nicht wo umherirrenden
    Gro�sohn hatte, gab ihm, was ihre Armut vermochte, und obendrein,
    was mehr war f�r Oliver und ihn alle seine Leiden auf eine Zeitlang
    vergessen lie�, freundliche Worte und mitleidige Z�hren.

    Am siebenten Morgen nach Sonnenaufgang erreichte er mit wunden F��en
    die kleine Stadt Varnet. Die Fensterl�den waren geschlossen, die
    Stra�en waren leer; nicht eine einzige Seele hatte sich schon zu den
    Gesch�ften des Tages erhoben. Die Sonne ging in all ihrer strahlenden
    Sch�nheit auf; aber ihr Licht diente nur dazu, dem Knaben seine
    Verlassenheit so recht zu Gem�te zu f�hren, als er mit blutenden F��en
    und staubbedeckt auf einer T�rschwelle sa�.

    Allm�hlich wurden die L�den ge�ffnet und die Rouleaus in die H�he
    gezogen, und die Leute begannen auf und ab zu gehen. Einige blieben
    stehen, um Oliver ein paar Augenblicke zu betrachten, oder wandten
    sich im Vorbeieilen um, um einen Blick auf ihn zu werfen; aber niemand
    k�mmerte sich um ihn oder fragte, wie er dorthin k�me. Er hatte nicht
    den Mut, jemand um eine Gabe anzusprechen. Nach einiger Zeit ging ein
    Knabe an ihm vor�ber, sah sich nach ihm um, ging weiter, sah sich noch
    einmal um, stand still, kehrte zur�ck und redete ihn an.

    Er mochte ungef�hr so alt sein wie Oliver selbst, der nie einen so
    absonderlichen Kauz gesehen. Er hatte eine Stumpfnase und eine
    platte Stirn, sah h�chst ordin�r und schmutzig aus, und seine ganze
    Haltung und sein Benehmen war wie das eines Mannes. Er war klein f�r
    sein Alter, hatte Dachsbeine und kleine, scharfe, h��liche Augen.
    Der Hut sa� ihm so lose auf dem Kopfe, als wenn er jeden Augenblick
    herunterfallen m��te, und er w�rde auch heruntergefallen sein, wenn er
    nicht durch h�ufige rasche Kopfbewegungen seines Besitzers immer wieder
    zurechtger�ckt oder befestigt worden w�re. Die Kleidung des Kleinen war
    gleichfalls nichts weniger als knabenhaft, und die ganze Figur stellte
    das vollkommene Bild eines renommierenden, prahlhaften kleinen Helden
    von vier Fu� H�he dar.

    �Was fehlt dir, Bursch? Was scheft dermehr?�[B] redete er Oliver an.

      [B] Was gibt's?

    �Ich bin sehr hungrig und m�de�, erwiderte Oliver, mit Tr�nen in den
    Augen. �Ich komme weit her und bin seit sieben Tagen auf der Wanderung
    gewesen.�

    �Weit her -- hm! -- seit sieben Tagen auf der Wanderung gewesen? --
    Ah -- sehe schon -- auf Oberschenkels Befehl -- he? Doch,� f�gte er
    hinzu, als er Olivers verwunderte Miene gewahrte, �du scheinst nicht zu
    wissen, was � Oberschenkel ist, mein guter Kochemer[C].�

      [C] Spitzbubenkamerad.

    Oliver erwiderte sch�chtern, er wisse allerdings sehr wohl, da� man
    unter einem Oberschenkel den oberen Teil eines Beines verstehe.

    �Ha, ha, ha! Wie gr�n!� rief der junge Gentleman aus. �� Oberschenkel
    ist � Friedensrichter, wer auf 'nes Oberschenkels Befehl geht, kommt
    nicht vorw�rts, sondern geht immer 'nauf, ohne wieder 'runter zu
    kommen. Noch nicht in der M�hle gewesen?�

    �In was f�r einer M�hle?� fragte Oliver.

    �Ei, in der, die in � Doves[D] Platz hat. Doch du bist butterich[E];
    ich hab' freilich auch nicht eben zu viel Massumme[F], aber so weit's
    zureicht, will ich rausr�cken und blechen. Steh auf -- komm!�

      [D] Gef�ngnis.

      [E] Hungrig.

      [F] Geld.

    Der junge Gentleman half Oliver aufstehen und nahm ihn mit sich in sein
    Gasthaus, wo er Brot und Schinken bringen lie� und ihn sehr aufmerksam
    beim Essen beobachtete. Als sich Oliver endlich ges�ttigt, warf er die
    Frage hin: �Nach London?�

    �Ja.�

    �Hast du eine Wohnung?�

    �Nein.�

    �Geld?�

    �Nein.�

    Der junge Herr senkte die H�nde in die Taschen und pfiff. --

    �Wohnst du in London?� fragte Oliver.

    �Ja, wenn ich zu Hause bin. Aber du wei�t wohl nicht, wo du kommende
    Nacht schlafen sollst?�

    �Nein�, antwortete Oliver. �Ich habe seit sieben N�chten unter keinem
    Dache geschlafen.�

    �Mach dir darum nur keine Sorgen. Ich gehe heute abend nach London und
    kenne da 'nen respektablen alten Herrn, der dir Wohnung umsonst geben
    und dir bald 'ne gute Stelle verschaffen wird -- das hei�t, wenn dich �
    Schentleman einf�hrt, den er kennt. Und ob er mich wohl kennt!� f�gte
    der junge Herr l�chelnd hinzu.

    Das unerwartete Anerbieten war zu lockend, als da� Oliver einen
    Augenblick h�tte anstehen sollen, es anzunehmen. Er wurde zutraulicher
    und erfuhr nun auch, da� sein neuer Freund Jack Dawkins hei�e und ein
    besonderer Liebling des erw�hnten alten Herrn sei. -- Jacks �u�eres
    schien freilich den Lieblingen des alten Herrn nicht viele Vorteile zu
    versprechen; allein da er ziemlich leichtfertig und gro�sprecherisch
    redete und auch gestand, da� er unter seinen Bekannten allgemein den
    Namen des �gepfefferten Baldowerers� (d. h. gewitzten Kundschafters)
    f�hre, so schlo� Oliver, er m�ge nicht eben viel taugen und die guten
    Lehren seines Wohlt�ters in den Wind schlagen. Oliver nahm sich daher
    in der Stille vor, sich so bald wie m�glich die Gunst des alten Herrn
    zu erwerben, und wenn er den Baldowerer unverbesserlich f�nde, die
    Ehre der n�heren Bekanntschaft mit ihm abzulehnen.

    Da es Jack nicht genehm war, vor Abend in London einzutreffen, so wurde
    es fast elf Uhr, bevor sie den Schlagbaum von Islington erreichten.
    Der Baldowerer f�hrte Oliver eiligen Schrittes durch ein Gewirr von
    Stra�en und Gassen, so da� sein Begleiter ihm kaum zu folgen vermochte.
    Trotz dieser Eile konnte Oliver nicht umhin, beim Weitergehen ein paar
    hastige Blicke nach beiden Seiten zu werfen. Eine schmutzigere oder
    elendere Gegend hatte er noch nie gesehen. Die Stra�en waren �u�erst
    eng und unsauber, und die Luft war mit �blen Ger�chen erf�llt. Es war
    eine gro�e Menge kleiner L�den vorhanden, aber der einzige Warenvorrat
    schien in Haufen von Kindern zu bestehen, die selbst zu dieser sp�ten
    Nachtstunde innerhalb und au�erhalb der T�ren umherkrochen oder im
    Innern der H�user schrien. Bedeckte Wege und H�fe, die hier und da von
    der Hauptstra�e abbogen, f�hrten zu kleinen H�usergruppen, vor denen
    betrunkene M�nner und Frauen sich tats�chlich im Schmutze w�lzten,
    und an verschiedenen Torwegen tauchten gro�gewachsene, verd�chtig
    aussehende Burschen auf, die allem Anschein nach nicht viel Gutes im
    Schilde f�hrten. Oliver �berlegte schon, ob er nicht am besten t�te,
    davonzulaufen, als ihn sein F�hrer pl�tzlich beim Arm nahm, die T�r
    eines Hauses unweit Fieldlane �ffnete, ihn hineinzog und die T�r wieder
    verschlo�. Der Baldowerer pfiff und erwiderte auf den Ruf: �Wer da?�
    -- �Grim und petacht!�[G] Unten auf dem Hausflur zeigte sich Licht,
    und der Kopf eines Mannes tauchte auf der zur K�che hinunterf�hrenden
    Treppe empor.

      [G] Gut und sicher.

    �Es sind euer zwei -- wer ist der andere?�

    �Ein neuer Chawwer�, rief Jack, Oliver nachziehend, zur�ck.

    �Woher kommt er?�

    �Von Gr�nland. Ist Fagin oben?�

    �Ja. Er sortiert die Schneichen[H]. Geh hinauf!�

      [H] Seidene T�cher.

    Das Licht wurde zur�ckgezogen, und der Kopf verschwand.

    Jack f�hrte Oliver eine finstere, sehr schadhafte Treppe hinauf, mit
    der er jedoch sehr genau bekannt zu sein schien, �ffnete die T�r eines
    Hinterzimmers und zog Oliver nach.

    Die W�nde des Gemachs waren von Schmutz und Rauch geschw�rzt, auf einem
    elenden Tische stand ein in den Hals einer Bierflasche gestecktes Licht
    und am Kamine die zusammengeschrumpfte Gestalt eines alten Juden mit
    einem zur�cksto�enden, spitzb�bischen, satanischen Gesicht, das durch
    dichte, klebrige, rote Haare verdunkelt wurde. Er steckte in einem
    fettigen flanellenen Schlafrocke, trug den Hals blo� und schien seine
    Aufmerksamkeit zwischen dem Feuer, an welchem er Brotschnitte r�stete,
    und dem Kleidergestell zu teilen, auf welchem eine gro�e Anzahl
    seidener Taschent�cher hing. An dem Tische sa�en vier oder f�nf Knaben,
    keiner �lter als Jack, rauchten aus langen Tonpfeifen und tranken
    Branntwein, ganz als wenn sie Erwachsene gewesen w�ren. Sie dr�ngten
    sich um den Baldowerer, als er dem Juden einige Worte zufl�sterte,
    drehten sich darauf nach Oliver um, und sie und der Jude grinsten ihn
    an.

    �Fagin, das ist er, mein Freund Oliver Twist�, sagte Jack Dawkins laut.

    Der Jude grinste, machte Oliver eine tiefe Verbeugung, fa�te seine
    Hand und sagte, er hoffe, die Ehre seiner n�heren Bekanntschaft zu
    haben. Hierauf umringten ihn die jungen, rauchenden Gentlemen und
    dr�ckten ihm eifrig die H�nde -- besonders die linke, in welcher er
    sein kleines B�ndel trug. Der eine von ihnen zeigte gro�en Eifer,
    seine Kappe aufzuh�ngen, und ein anderer war so dienstfertig, in
    seine Tasche zu greifen, um ihn der M�he zu �berheben, wenn er sich
    niederlegte, sie auszuleeren; und alle diese H�flichkeiten w�rden kein
    Ende gehabt haben, wenn der Jude die K�pfe und Schultern der gef�lligen
    jungen Herren nicht mit der R�stgabel, die er in der Hand hielt, zu
    bearbeiten angefangen h�tte.

    �Wir sind alle sehr erfreut, dich kennen zu lernen, Oliver�, sagte
    der Jude. �Baldowerer, mache einen Platz f�r Oliver am Feuer frei.
    Ah, du betrachtest verwundert die Taschent�cher, mein Lieber? Nicht
    wahr, es sind ihrer eine ganze Menge? Wir haben sie soeben zum Waschen
    herausgeh�ngt. Das ist alles, Oliver; das ist alles. Ha, ha, ha!�

    Seine letzten Worte wurden von einem schallenden Gel�chter all der
    hoffnungsvollen Z�glinge des lustigen alten Herrn begr��t, worauf sich
    alle zu Tisch setzten.

    Nachdem Oliver seinen Teil gegessen, mischte ihm der Jude ein Glas
    hei�en Genever mit Wasser und sagte ihm, er m�sse sogleich austrinken,
    weil noch jemand des Glases bed�rfe. Oliver tat, was ihm gehei�en
    war, sein Freund Jack hob ihn auf, legte ihn auf ein aus alten S�cken
    bereitetes Lager, und er versank sogleich in einen tiefen Schlummer.




    9. Kapitel.

        Weitere Mitteilungen �ber den alten Herrn und seine hoffnungsvollen
        Z�glinge.


    Es war schon sp�t am folgenden Morgen, als Oliver aus einem langen,
    festen Schlummer erwachte, doch vorerst nur zu jenem Mittelzustande
    zwischen Schlaf und Wachen, in welchem man sich noch nicht vollkommen
    ermuntern kann und doch alles h�rt und sieht, was umher vorgeht.

    Der Jude war au�er Oliver allein im Zimmer. Er schl�rfte seinen
    Kaffee, setzte das Geschirr nach einiger Zeit zur Seite, stand eine
    Weile am Kamin, wie wenn er nicht w��te, was er zun�chst vornehmen
    sollte, blickte darauf nach Oliver hin und rief ihn beim Namen. Oliver
    antwortete nicht und schien noch zu schlafen.

    Der Jude horchte, ging zur T�r, schob den Riegel vor und nahm darauf,
    wie es Oliver schien, aus einer Vertiefung des Fu�bodens eine kleine
    Schachtel heraus und stellte sie auf den Tisch. Seine Augen gl�nzten,
    als er sie �ffnete und in die Schachtel hineinschaute. Er setzte sich
    und nahm eine goldene, von Diamanten funkelnde Uhr heraus.

    �Aha!� murmelte er mit einem entsetzlichen L�cheln. �Verdammt pfiffige
    Bestien! Und courageux bis zum letzten Augenblick. Sagten mit keinem
    Sterbensw�rtchen dem alten Pfarrer, wo sie w�ren, verkappten[I] den
    alten Fagin nicht. Und was h�tt's ihnen geholfen? Der Strick w�re doch
    geblieben fest -- h�tten gebaumelt keinen Augenblick sp�ter. Nein,
    nein! Wackre Bursche, wackre Bursche!�

      [I] Verraten.

    Er legte die Uhr wieder in die Schachtel, nahm mehrere andere und dann
    Ringe, Armb�nder und viele Kostbarkeiten heraus, deren Namen oder
    Gebrauch Oliver nicht einmal kannte, und be�ugelte sie mit gleichem
    Vergn�gen. Hierauf legte er ein sehr kleines Geschmeide in seine
    flache Hand und schien lange bem�ht, zu lesen, was darin eingegraben
    sein mochte. Endlich lie� er es, wie am Erfolge verzweifelnd, wieder
    in die Schachtel hineinfallen, lehnte sich zur�ck und murmelte: �Was
    es doch ist f�r 'ne h�bsche Sache ums H�ngen! Tote bereuen nicht --
    bringen ans Licht keine dummen Geschichten. Selbst die Aussicht auf den
    Galgen macht sie keck und dreist. 's ist sehr sch�n f�rs Gesch�ft. F�nf
    aufgehangen in einer Reihe, und keiner �brig zu teilen mit mir oder zu
    lehmern[J].�

      [J] Verraten, beichten.

    Er blickte auf, seine schwarzen, stechenden Augen begegneten Olivers
    Blicken, die in stummer Neugier auf ihn geheftet waren, und er gewahrte
    sogleich, da� er beobachtet worden war. Er dr�ckte die Schachtel zu,
    griff nach einem auf dem Tische liegenden Messer und sprang w�tend und
    am ganzen Leibe zitternd auf.

    �Was ist das?� rief er. �Warum passest du mir auf? Warum bist du wach?
    Was hast du gesehen? Sprich, Bube -- sprich, sprich, so lieb dir dein
    Leben ist!�

    �Ich konnte nicht mehr schlafen�, erwiderte Oliver best�rzt. �Es tut
    mir sehr leid, wenn ich Sie gest�rt habe, Sir!�

    �Hast du nicht schon seit einer Stunde gewacht?� fragte der Jude,
    Oliver finster anblickend.

    �Nein, Sir -- nein, wahrlich nicht�, sagte Oliver.

    �Ist's auch wahr?� rief der Jude mit noch drohenderen Geb�rden.

    �Auf mein Wort, Sir!� versicherte Oliver.

    �Schon gut, schon gut!� fuhr der Jude, auf einmal sein gew�hnliches
    Wesen wieder annehmend, fort. �Ich wei� es wohl -- wollte dich nur
    erschrecken -- auf die Probe stellen. Du bist ein wackerer Junge,
    Oliver.� Er rieb sich kichernd die H�nde, blickte jedoch unruhig nach
    der Schachtel hin. �Hast du gesehen die h�bschen Sachen?� fragte er
    nach einigem Stillschweigen.

    �Ja, Sir.�

    �Ah!� rief erblassend der Jude aus. �Sie -- sind mein Eigentum, Oliver;
    mein kleines Eigentum -- alles, was ich besitze f�r meine alten Tage.
    Man schilt mich einen Geizhals -- aber ich mu� doch leben.�

    Oliver dachte, der alte Herr m�sse wirklich ein Geizhals sein, denn er
    w�rde sonst nicht, obgleich im Besitz solcher Sch�tze, so erb�rmlich
    wohnen. Indes meinte er, seine Liebe zu Jack und den anderen Knaben
    m�chte ihm wohl viel Geld kosten. Er fragte sch�chtern, ob er aufstehen
    d�rfe. Der Jude hie� ihn Wasser zum Waschen aus dem dastehenden
    Steinkruge holen, und als Oliver es gesch�pft hatte und sich umdrehte,
    war die Schachtel verschwunden.

    Er hatte sich kaum gewaschen, als der Baldowerer nebst einem der Knaben
    eintrat, die Oliver am vorigen Abend hatte rauchen sehen. Jack stellte
    ihm seinen Begleiter, Charley Bates, f�rmlich vor, und alle vier
    setzten sich zum Fr�hst�ck, das Jack in seinem Hute mitgebracht hatte.

    �Ich hoffe, da� ihr heute morgen gearbeitet habt!� sagte der Jude zu
    Jack, nach Oliver blinzelnd.

    �T�chtig!� lautete die Antwort.

    �Wie Drescher!� setzte Charley Bates hinzu.

    �Ah, ihr seid gute Jungen! Was hast du mitgebracht, Baldowerer?�

    �Ein paar Brieftaschen!� erwiderte Jack und reichte ihm eine rote und
    eine gr�ne hin.

    Der Jude �ffnete beide und durchsuchte sie mit bebender Begier. �Nicht
    so schwer, als sie sein k�nnten�, bemerkte er; �aber doch artige
    Arbeit, recht artige Arbeit -- nicht wahr, Oliver?�

    �Ja, wahrlich, Sir!� antwortete Oliver, wor�ber Charley Bates, zur
    gro�en Verwunderung Olivers, laut zu lachen anfing.

    �Was hast du denn mitgebracht, Charley?� fragte der Jude.

    �Schneichen!� erwiderte Master Bates und wies vier Taschent�cher vor.

    Der Jude nahm sie in genauen Augenschein.

    �Sie sind sehr gut�, sagte er; �du hast sie aber nicht gezeichnet gut;
    die Buchstaben m�ssen wieder ausgel�st werden, und das soll Oliver
    lernen. Willst du, Oliver?�

    �Wenn Sie es befehlen, gern, Sir!� war Olivers Antwort.

    �M�chtest du mir wohl ebenso leicht Taschent�cher anschaffen k�nnen wie
    Charley?�

    �Warum nicht -- wenn Sie es mich lehren wollen, Sir?�

    Charley brach abermals in ein schallendes Gel�chter aus und w�re dabei
    fast erstickt, da er eben einen Bissen zum Munde gef�hrt hatte. �Er ist
    gar zu allerliebst gr�n!� rief er endlich, gleichsam zur Entschuldigung
    seines unh�flichen Benehmens, aus.

    Der Baldowerer bemerkte, Oliver w�rde seinerzeit schon alles lernen.
    Der Jude sah Oliver die Farbe wechseln und lenkte das Gespr�ch auf
    einen anderen Gegenstand. Er fragte, ob viele Zuschauer bei der
    Hinrichtung gewesen w�ren, und Olivers Erstaunen wuchs immer mehr,
    denn aus den Antworten Jacks und Charleys ging hervor, da� sie
    beide zugegen gewesen waren, und es war ihm unerkl�rlich, wie sie
    dessenungeachtet so flei�ig hatten arbeiten k�nnen.

    Als das Fr�hst�ck beendet war, spielten der muntere alte Herr und die
    beiden Knaben ein �u�erst sonderbares und ungew�hnliches Spiel. Der
    alte Herr steckte eine Dose, eine Brieftasche und eine Uhr in seine
    Taschen, eine Brustnadel in sein Hemd, hing eine Uhrkette um den Hals,
    kn�pfte den Rock dicht zu, ging auf und ab, blieb bisweilen stehen,
    als wenn er in einen Laden hineins�he, blickte best�ndig umher, als
    wenn er Furcht vor Dieben hegte, bef�hlte seine Taschen, wie um sich
    zu �berzeugen, ob er auch nichts verloren h�tte, und machte das alles
    so spa�haft und nat�rlich, da� Oliver lachte, bis ihm die Tr�nen �ber
    die Wangen hinabliefen. Die beiden Knaben verfolgten unterdes den
    Alten und entschwanden, wenn er sich umdrehte, seinen Blicken mit der
    bewunderungsw�rdigsten Behendigkeit. Endlich trat ihm der Baldowerer
    wie zuf�llig auf die Zehen, w�hrend Charley Bates von hinten gegen ihn
    anrannte, und sie entwendeten ihm dabei Taschentuch, Uhr, Brustnadel
    usw. so geschickt, da� Oliver kaum ihren Bewegungen zu folgen
    vermochte. F�hlte der alte Herr eine Hand in einer seiner Taschen, so
    war der Dieb gefangen, und das Spiel fing von vorn wieder an.

    Es war mehreremal durchgespielt, als zwei junge Damen erschienen, um
    die jungen Herren zu besuchen. Die eine hie� Betsy, die andere Nancy.
    Ihr Haar war nicht in der genauesten Ordnung, ihre Schuhe und Str�mpfe
    schienen nicht im besten Zustande zu sein. Sie waren vielleicht nicht
    eigentlich sch�n, hatten aber viel Farbe und ein kr�ftiges, munteres
    Aussehen. Ihre Manieren waren sehr frei und angenehm, und so meinte
    Oliver, da� sie sehr artige M�dchen w�ren, was sie auch ohne Zweifel
    waren.

    Sie blieben lange. Es wurden geistige Getr�nke gebracht, da die jungen
    Damen �ber innerliche K�lte klagten, und die munterste Unterhaltung
    entspann sich. Endlich erinnerte sich Charley Bates, da� es Zeit sei,
    auszugehen. Der gute alte Herr gab ihm und dem Baldowerer verschiedene
    Anweisungen und Geld zum Ausgeben, worauf sie sich nebst Betsy und
    Nancy entfernten.

    �Ist's nicht ein angenehmes Leben, das meine Knaben f�hren?� sagte
    Fagin.

    �Sind sie denn auf Arbeit ausgegangen?� fragte Oliver.

    �Allerdings�, erwiderte der Jude; �und sie arbeiten den ganzen Tag
    unverdrossen, wenn sie nicht werden gest�rt. Nimm sie dir zum Muster,
    mein Kind; tu alles, was sie dir hei�en; und folg' jederzeit ihrem Rat,
    besonders dem des Baldowerers. Er wird werden ein gro�er Mann und auch
    aus dir machen 'nen gro�en Mann, wenn du dir ihn zum Vorbilde nimmst.
    H�ngt mein Taschentuch aus der Tasche, mein Lieber?�

    �Ja, Sir!� sagte Oliver.

    �So sieh einmal zu, ob du es herausziehen kannst, ohne da� ich's f�hle,
    wie du's vorhin gesehen hast von den beiden.�

    Oliver erinnerte sich genau, wie er es Jack hatte tun sehen, und tat es
    ihm nach.

    �Ist's heraus?�

    �Hier ist es, Sir.�

    �Du bist ein kluger Knabe�, sagte der alte Herr, ihm die Wange
    klopfend; �ich habe niemals gesehen ein anstelligeres Kind. Da hast du
    'nen Schilling. F�hrst du so fort, so wirst du werden der gr��te Mann
    deiner Zeit. Doch will ich dir jetzt zeigen, wie man herausl�st die
    Buchstaben.�

    Oliver konnte gar nicht begreifen, wie er ein gro�er Mann dadurch
    werden k�nne, da� er dem alten Herrn das Tuch aus der Tasche z�ge,
    meinte jedoch, da� es der so viel �ltere besser wissen m�sse als er,
    und war bald eifrig mit seinen neuen Studien besch�ftigt.




    10. Kapitel.

        Oliver gewinnt Erfahrung um einen hohen Preis.


    Oliver blieb acht bis zehn Tage im Zimmer des Juden, wurde fortw�hrend
    besch�ftigt, Zeichen aus den Taschent�chern, von denen eine gro�e Menge
    nach Hause gebracht wurde, herauszutrennen, und nahm bisweilen an dem
    beschriebenen Spiele teil, das t�glich gespielt wurde. Er fing immer
    mehr an, sich nach frischer Luft zu sehnen, und bat den alten Herrn
    mehrmals auf das dringendste, ihn mit seinen beiden Kameraden zum
    Arbeiten ausgehen zu lassen.

    Endlich wurde ihm eines Morgens die Erlaubnis erteilt, unter Jacks und
    Charleys Aufsicht auszugehen. Es waren keine Taschent�cher mehr da,
    an denen Oliver h�tte arbeiten k�nnen, und vielleicht war dies der
    Grund, weshalb der alte Herr seine Zustimmung gab. Die Knaben gingen
    und gerieten sogleich in ein sehr langsames Schlendern, was Oliver
    h�chst mi�billigte, eingedenk der vielfachen Warnungen des alten Herrn
    vor dem verderblichen M��iggange. Der Baldowerer ver�bte mannigfachen
    Mutwillen an Knaben, und Charley erlaubte sich sogar, die Heiligkeit
    des Eigentums zu verletzen, wenn er an einem Apfel- oder Zwiebelkorbe
    vor�berkam. Oliver war daher schon im Begriff, unwillig heimzukehren,
    als seine Begleiter auf einmal anfingen, sich �u�erst geheimnisvoll zu
    benehmen, wodurch er von seinem Vorhaben abgelenkt wurde.

    Sie umschlichen einen alten Herrn, auf den sie ihn aufmerksam gemacht
    hatten, ohne seine Fragen anders als durch einige ihm unverst�ndliche
    Worte und Winke zu beantworten. Er hielt sich einige Schritte hinter
    ihnen und stand endlich, unschl�ssig, ob er weitergehen oder sich
    zur�ckziehen solle, verwundert zuschauend da.

    Der alte Herr sah sehr respektabel aus, trug Puder in den Haaren und
    eine goldene Brille. Er hatte sich vor einen B�cherladen hingestellt,
    ein Buch zur Hand genommen, las darin, sein spanisches Rohr unter dem
    linken Arme, und h�rte und sah offenbar nicht, was um ihn her vorging.

    Wer beschreibt Olivers Best�rzung, als der Baldowerer dem alten Herrn
    das Tuch aus der Tasche zog, es Charley Bates reichte, und als darauf
    beide spornstreichs davonliefen! Im Augenblick war ihm das Geheimnis
    der Taschent�cher, Uhren und Kleinodien klar. Das Blut stockte ihm
    in den Adern, ihm schwindelte vor Furcht und Schrecken, und ohne zu
    wissen, was er tat, lief er seinen Kameraden nach, so schnell seine
    F��e ihn tragen mochten. In demselben Augenblick griff der alte Herr
    nach seinem Tuche in die Tasche, vermi�te es, drehte sich rasch um, sah
    Oliver laufen und erhob den Ruf: �Halt den Dieb!� -- den magischen Ruf,
    auf welchen sofort alles lebendig wird, der Kr�mer aus seinem Laden
    auf die Stra�e st�rzt, der Gem�seh�ndler seinen Korb, der Milchmann
    seinen Eimer, der Pflasterer seine Ramme, der Schulknabe seine B�cher
    im Stiche l��t und alles nachl�uft.

    Jack und Charley hatten Aufsehen zu vermeiden gew�nscht und waren
    daher nur bis um die n�chste Ecke gelaufen, worauf sie sich unter
    einem Torwege neugierigen Blicken zu entziehen suchten. Sobald sie das
    Geschrei �Halt den Dieb!� vernahmen, stimmten sie aus allen Kr�ften ein
    und schlossen sich wie gute B�rger den Verfolgern an. Diese Anwendung
    des gro�en Naturgesetzes der Selbsterhaltung war Oliver vollkommen neu.
    Er wurde noch mehr verwirrt und best�rzt und verdoppelte seine Eile,
    sah sich indes nach einiger Zeit eingeholt und wurde obendrein zu Boden
    geschlagen.

    In wenigen Augenblicken war ein zahlreicher Haufen um ihn versammelt.
    �Dr�ckt ihn doch nicht tot!� -- �Verdient er's besser?� -- �Wo ist der
    bestohlene Herr?� -- �Da kommt er schon; macht Raum f�r den Herrn!� --
    �Ist dies der Bursch, Sir?� -- �Ja!�

    Oliver lag da, mit Schmutz bedeckt, blutend aus Nase und Mund, und sah
    bet�ubt und ge�ngstet umher.

    �Ich f�rchte, da� es der Knabe ist�, sagte der Herr sehr milde.

    �Das f�rchten Sie? Der ist auch wohl der Rechte.�

    �Der arme Kleine hat sich besch�digt!� fuhr der Herr fort.

    �Das hab' ich getan�, fiel ein vierschr�tiger Mensch, hervortretend,
    ein; �traf ihn gerade mit der Faust auf die Schnauze -- ich hab' ihn
    aufgehalten f�r Sie, Sir.�

    Er zog grinsend den Hut, eine Belohnung seiner Dienstfertigkeit
    erwartend; allein der alte, dicke Herr blickte ihn unwillig an und
    h�tte sich offenbar gern entfernt, wenn sich nicht ein Polizist, der in
    solchen F�llen gew�hnlich zuletzt kommt, in diesem Augenblick durch die
    Menge gedr�ngt und Oliver beim Kragen gepackt h�tte.

    �Steh auf!� sagte der Mann barsch.

    �Ich bin es wirklich nicht gewesen, Sir, wirklich und wahrhaftig
    nicht. Es waren zwei andere Knaben�, sagte Oliver, die H�nde bittend
    zusammenlegend. �Sie m�ssen hier irgendwo in der N�he sein.�

    �O nein, sie sind nicht hier�, entgegnete der Beamte. Er meinte dies
    ironisch, aber es war die volle Wahrheit, denn der Baldowerer und
    Charley Bates hatten sich l�ngst aus dem Staube gemacht. �Steh auf!�

    �Tun Sie ihm nichts zuleide�, sagte der menschenfreundliche Herr.

    �O nein, ich werde ihm nichts zuleide tun�, erwiderte der Polizist,
    indem er zum Beweise daf�r Oliver die Jacke halb vom R�cken ri�. �Komm
    nur; ich kenne dich schon. Willst du mal auf deinen F��en stehen,
    verdammter kleiner Strolch!�

    Oliver machte einen Versuch, sich zu erheben, konnte sich aber kaum
    aufrecht erhalten und wurde am Kragen seiner Jacke im Laufschritt
    durch die Stra�en geschleppt. Der alte Herr ging mit, und ein immer
    anwachsender Volkshaufen folgte johlend und l�rmend den drei nach der
    n�chsten Polizeiwache.




    11. Kapitel.

        Wie Mr. Fang die Gerechtigkeit handhabte.


    Der Diebstahl war im Bezirke dieses Polizeiamtes begangen worden.
    Als der Zug auf der Wache anlangte, wurde Oliver vorl�ufig in ein
    kellerartiges Gemach eingeschlossen, das �ber alle Beschreibung
    schmutzig war, denn sechs Betrunkene hatten es fast drei Tage
    inne gehabt. Doch das will nichts sagen. Sperrt man doch Tag f�r
    Tag und Nacht f�r Nacht M�nner und Weiber um der geringf�gigsten,
    leichtfertigsten Anschuldigungen willen in Spelunken ein, gegen welche
    die Zellen der schwersten und bereits verurteilten Verbrecher im
    Newgategef�ngnisse f�r Prunkgem�cher gelten k�nnten!

    Der alte Herr sah Oliver mitleidig und wehm�tig nach. --

    �Es liegt ein Ausdruck in den Z�gen des Knaben, der mich ganz wunderbar
    ergreift�, sprach er bei sich selbst. �Sollte er nicht unschuldig sein?
    Er sah aus, als wenn er -- hm! -- ist mir's doch in der Tat, als wenn
    ich dieses Gesicht oder ein ganz �hnliches schon gesehen h�tte.�

    Er sann und sann, rief sich die Z�ge seiner Freunde, Feinde und
    Bekannten, alter und neuer, l�ngst vergessener, l�ngst im Grabe
    ruhender ins Ged�chtnis zur�ck, vermochte sich aber dennoch auf keines
    zu entsinnen, mit welchem Oliver �hnlichkeit gehabt h�tte. �Nein, es
    mu� Einbildung sein�, sagte er endlich seufzend und kopfsch�ttelnd.

    Er wurde durch eine Ber�hrung an der Schulter aus seinem Sinnen
    aufgeschreckt und bemerkte, als er sich umwandte, den Schlie�er, der
    ihn aufforderte, ihm ins Amtszimmer zu folgen. Als er eintrat, sa� Mr.
    Fang, der Polizeirichter, bereits hinter einer Barriere am oberen Ende,
    und neben der T�r befand sich eine Art von h�lzernem Verschlag, in dem
    der arme Oliver, an allen Gliedern zitternd, hockte. Mr. Fangs Antlitz
    hatte den Ausdruck der H�rte und war sehr rot. Wenn er nicht mehr zu
    trinken pflegte, als ihm gut war, so h�tte er gegen sein Gesicht eine
    Injurienklage anstellen k�nnen, und sicher w�rden ihm betr�chtliche
    Entsch�digungsgelder zuerkannt worden sein.

    Der alte Herr verbeugte sich ehrerbietig.

    �Hier ist mein Name und meine Adresse, Sir!� sagte er und reichte Mr.
    Fang seine Karte.

    Mr. Fang, der eben seine Zeitung las, war unwillig �ber die St�rung und
    blickte �rgerlich auf.

    �Wer sind Sie?�

    Der alte Herr wies ein wenig erstaunt auf seine Karte.

    Mr. Fang stie� sein Zeitungsblatt nebst der Karte ver�chtlich zur Seite.

    �Gerichtsdiener! Wer ist dieser Mensch?�

    �Sir, ich hei�e Brownlow�, fiel der alte Herr mit dem Anstande eines
    Gentleman in starkem Kontrast zu Mr. Fang ein. �Erlauben Sie, da� ich
    um den Namen des Richters bitte, der einen anst�ndigen Mann ohne alle
    Veranlassung im Gerichtslokale beleidigt.�

    �Gerichtsdiener!� herrschte Fang; �wessen ist dieser Mensch angeklagt?�

    �Er ist nicht angeklagt, Ihr Edeln, sondern erscheint als Ankl�ger des
    Knaben.�

    Seine Edeln wu�ten das sehr wohl, konnten jedoch auf die Weise ganz
    sicher unangenehme Dinge sagen.

    �Erscheint als Ankl�ger des Knaben -- so!� sagte Fang, Brownlow
    ver�chtlich von Kopf bis zu den F��en betrachtend. �Nehmen Sie ihm den
    Eid ab.�

    �Bevor das geschieht, mu� ich mir ein paar Worte erlauben�, fiel
    Brownlow ein. �Ich w�rde n�mlich, ohne da� es mir wirklich widerfahren
    w�re, niemals geglaubt haben --�

    �Halten Sie den Mund, Sir!� unterbrach ihn Fang in befehlshaberischem
    Tone.

    �Ich will und werde reden!� sagte Brownlow ebenso bestimmt.

    �Sie halten augenblicklich den Mund, Sir, oder ich lasse Sie
    hinausbringen. Sie sind ein unversch�mter Mensch! Wie k�nnen Sie es
    wagen, sich den Anordnungen eines Richters widersetzen zu wollen?�

    Dem alten Herrn stieg das Blut ins Gesicht.

    �Vereidigen Sie dieses Individuum!� rief Fang dem Schreiber zu. �Ich
    will durchaus nichts mehr h�ren.�

    Brownlow war im h�chsten Grade entr�stet, glaubte aber, dem Knaben
    m�glicherweise schaden zu k�nnen, wenn er seine Gef�hle nicht
    unterdr�ckte, und legte daher den Eid ab.

    �Wohin geht Ihre Anklage?� fragte ihn Fang darauf. �Was haben Sie zu
    sagen, Sir?�

    �Ich stand vor einem B�cherladen�, begann Brownlow, allein Fang
    unterbrach ihn.

    �Schweigen Sie, Sir. Wo ist der Polizist? Vereidigen Sie den
    Polizisten. Polizist -- reden Sie!�

    Der Polizist berichtete mit geb�hrender Unterw�rfigkeit, wie er den
    Knaben gefunden, und wie er ihm die Taschen durchsucht und nichts
    gefunden habe; -- mehr wisse er nicht.

    �Sind Zeugen vorhanden?� fragte Fang.

    �Nein, Ihr Edeln.�

    Fang sa� ein paar Minuten schweigend da, wendete sich darauf zu
    Brownlow und sagte in gro�er Hitze: �Denken Sie Ihre Anklage gegen den
    Knaben anzubringen oder nicht? Sie haben geschworen. Verweigern Sie Ihr
    Zeugnis, so werd' ich Sie wegen Nichtachtung der Richterbank in Strafe
    nehmen; das werd' ich, beim --�

    Es ist und bleibt unbekannt, bei wem; denn der Schreiber hustete im
    rechten Augenblick und lie� ein Buch zur Erde fallen -- nat�rlich nur
    zuf�llig.

    Brownlow konnte endlich vorbringen, was er zu sagen hatte, und f�gte
    hinzu, da� er die Hoffnung hege, der Richter werde die Gesetze so mild
    wie m�glich anwenden, wenn er es als erwiesen annehmen sollte, da� der
    Knabe, wenn er nicht selbst ein Dieb sei, doch mit Dieben in Verbindung
    stehe.

    �Er ist bereits hart besch�digt,� schlo� er, �und ich f�rchte, da� ihm
    sehr unwohl ist.�

    �Unwohl -- so, so!� sagte Fang mit einem h�hnischen L�cheln. �Du
    spielst mir hier keine Kom�die, du kleiner Landstreicher, das sag' ich
    dir; kommst mir damit nicht durch. Wie hei�est du?�

    Oliver wollte antworten, aber die Zunge versagte den Dienst. Er war
    totenbla�, und alles schien sich mit ihm zu drehen.

    �Wie hei�est du, du verh�rteter Schlingel?� donnerte ihn Fang
    wiederholt an. �Gerichtsdiener, wie hei�t der Bube?�

    Der Gerichtsdiener beugte sich �ber Oliver und wiederholte die Frage,
    gewahrte aber, da� der Knabe wirklich nicht imstande war zu antworten,
    und sagte daher, weil er wu�te, da� der Richter sonst nur noch w�tender
    werden und eine noch h�rtere Strafe diktieren w�rde: �Er sagt, sein
    Name w�re Tom White, Ihr Edeln.�

    �Wo wohnt er?� fragte Fang weiter.

    �Wo er eben kann!� erwiderte der gutherzige Gerichtsdiener abermals f�r
    Oliver.

    �Hat er Eltern?�

    �Er sagt, sie w�ren in seiner Kindheit gestorben, Ihr Edeln!�
    entgegnete der Gerichtsdiener. Es war die gew�hnliche Antwort in F�llen
    dieser Art.

    Oliver hob bei der letzten Frage den Kopf empor, sah mit flehenden
    Blicken umher und bat mit schwacher Stimme um ein Glas Wasser.

    �Albernheiten!� sagte Fang. �Hab' mich ja nicht zum Narren, Bursch!�

    �Ich glaube wirklich, da� ihm unwohl ist, Ihr Edeln!� wendete der
    Gerichtsdiener ein.

    �Ich wei� es besser�, fuhr Fang auf.

    �Gerichtsdiener, halten Sie ihn!� rief der alte Herr, �oder er sinkt zu
    Boden.�

    �Zur�ck da, Gerichtsdiener!� tobte Fang; �mag er, wenn's ihm beliebt.�

    Oliver bediente sich der freundlichen Erlaubnis und fiel ohnm�chtig von
    seiner Bank herunter.

    Der Richter befahl, ihn liegen zu lassen, bis er wieder zu sich k�me;
    der Schreiber fragte leise, wie Mr. Fang zu verfahren ged�chte.

    �Summarisch�, erwiderte Mr. Fang. �Er wird drei Monate eingesperrt --
    nat�rlich bei harter Arbeit.�

    Zwei Schlie�er schickten sich an, den ohnm�chtigen Knaben in seine
    Zelle zu tragen, als pl�tzlich ein �ltlicher, �rmlich, aber anst�ndig
    gekleideter Mann atemlos hereintrat.

    �Halt -- halt!� rief er; �um des Himmels willen noch einen Augenblick
    Geduld.�

    Obgleich die Polizeibeamten die willk�rlichste Gewalt �ber die
    Freiheit, den guten Ruf und Namen, ja fast das Leben der k�niglichen
    Untertanen, besonders der �rmeren Klassen, zu �ben pflegen, und
    obgleich in den Polizeigerichten genug Dinge vorgehen, um den Engeln
    blutige Tr�nen auszupressen, so erf�hrt das Publikum doch nichts davon,
    ausgenommen durch das Medium der Tagespresse. Mr. Fang war daher nicht
    wenig entr�stet, einen ungebetenen Gast eintreten und so ordnungswidrig
    auftreten zu sehen.

    �Was ist das? Wer ist das? Werft den Menschen hinaus!� rief er.

    �Ich will und mu� reden, Sir; ich lasse mich nicht hinauswerfen; hab's
    alles angesehen. Ich bin der Besitzer des Buchladens. Ich verlange,
    vereidigt zu werden. Mr. Fang, Sie m�ssen mich anh�ren -- Sie k�nnen es
    nicht wagen, mein Zeugnis zur�ckzuweisen, Sir.�

    Er war im Recht und sah zu entschlossen aus, als da� der Richter
    es h�tte wagen d�rfen, ihn abzuweisen. Fang lie� ihm daher den Eid
    abnehmen und fragte darauf, was er zu sagen habe.

    �Ich sah drei Knaben -- zwei andere und diesen hier -- um den Herrn
    da herumschleichen, der vor meinem Laden stand und las. Der Diebstahl
    wurde von einem anderen Knaben begangen, und dieser war ganz erstaunt
    dar�ber -- sah aus, als wenn ihn der Schlag ger�hrt h�tte.�

    �Warum kamen Sie nicht schon fr�her her?�

    �Ich hatte niemand, nach meinem Laden zu sehen, und bin hergelaufen,
    sobald ich jemand auftreiben konnte.�

    �Also der Ankl�ger las?�

    �Ja, Sir -- in dem Buche, das er in diesem Augenblicke in der Hand hat.�

    �Ah -- ist es bezahlt?�

    �Nein!� erwiderte der Buchh�ndler l�chelnd.

    �Mein Himmel, das hab' ich ganz vergessen!� rief der zerstreute alte
    Herr ganz unbefangen aus.

    �Vortrefflich! -- Und Sie werfen sich zum Ankl�ger eines ungl�cklichen,
    armen Knaben auf!� bemerkte Fang mit komisch aussehender Anstrengung,
    eine menschenfreundliche Miene anzunehmen. �Es scheint mir, Sir, da�
    Sie unter sehr verd�chtigen und unehrenhaften Umst�nden zu dem Buche
    gelangt sind, und Sie k�nnen sich sehr gl�cklich sch�tzen, wenn der
    Eigent�mer nicht als Ankl�ger gegen Sie auftreten will. Nehmen Sie
    sich dies zur Lehre, mein Freund, oder Sie verfallen noch einmal dem
    Gesetze. Der Knabe ist freizulassen. R�umen Sie das Gerichtszimmer!�

    Der alte Herr wurde unter Ausbr�chen der Entr�stung, die er nicht
    l�nger mehr zur�ckzuhalten vermochte, hinausgef�hrt. Er stand im
    Hofraume, und sein Zorn verschwand. Oliver lag auf dem Steinpflaster;
    man hatte ihm die Schl�fe mit Wasser gewaschen; er war wei� wie eine
    Leiche und zitterte krampfhaft am ganzen Leibe. �Armes Kind, armes
    Kind!� sagte Mr. Brownlow, sich �ber ihn hinunterbeugend. �Leute, ich
    bitte, schaff' mir doch jemand sogleich einen Mietwagen.�

    Gleich darauf fuhr ein leerer Wagen vor�ber, Oliver wurde sorgf�ltig
    hineingehoben und auf einen Sitz gelegt, w�hrend der alte Herr auf dem
    anderen Platz nahm.

    �Darf ich Sie begleiten?� fragte der Buchh�ndler.

    �Ja, ja, mein werter Herr!� erwiderte Brownlow. �Ich habe Sie
    vergessen; verzeihen Sie. Und da hab' ich auch das ungl�ckliche Buch
    noch. Steigen Sie geschwind ein, es ist keine Zeit zu verlieren.�

    Der Buchh�ndler setzte sich zu Brownlow, und sie fuhren ab.




    12. Kapitel.

        In welchem f�r Oliver bessere F�rsorge getragen wird, als er sie
        noch in seinem ganzen Leben erfahren. Die Geschichte kehrt zu dem
        lustigen alten Herrn und seinen hoffnungsvollen Z�glingen zur�ck.


    Der Wagen hielt nach ziemlich langer Fahrt vor einem h�bschen Hause in
    einer stillen Stra�e, nicht weit von Pentonville. Mr. Brownlow lie�
    Oliver sogleich zu Bett bringen und sorgte mit einem Eifer f�r Pflege
    jeder Art, der keine Grenzen kannte. Sein Sch�tzling verfiel in ein
    heftiges Fieber und erwachte erst nach acht Tagen aus einem langen
    und unruhigen Traume, wie es ihm schien. �Wo bin ich?� rief er mit
    schwacher Stimme. �Wer hat mich hierher gebracht?�

    Der Vorhang seines Bettes wurde rasch zur�ckgeschoben, und eine
    m�tterlich aussehende, sauber gekleidete alte Frau beugte sich �ber
    ihn und sagte: �Ruhig, mein S�hnchen, du mu�t ganz still liegen oder
    wirst sonst wieder krank werden. Denn du hast an der Schwelle des Todes
    gestanden; also verhalte dich ja recht ruhig.�

    Sie sah so freundlich und liebevoll dabei aus und strich ihm so
    sorglich das Haar von der Stirn zur�ck, da� er sich nicht enthalten
    konnte, seine abgezehrte Hand auf die ihrige zu legen und einige, wenn
    auch unverst�ndliche Worte ger�hrten Dankes zu murmeln.

    �Was es f�r ein lieber Kleiner ist!� sagte sie mit Tr�nen in den Augen.
    �Wie w�rde sich seine Mutter freuen, wenn sie so wie ich bei ihm
    gesessen h�tte und ihn jetzt s�he!�

    �Vielleicht sieht sie mich,� fl�sterte Oliver und faltete seine H�nde.
    �Vielleicht war sie bei mir, Ma'am. Es ist mir fast, als w�re sie hier
    gewesen.�

    �Das macht das Fieber, mein Kind�, bemerkte Frau Bedwin.

    �Kann wohl sein�, erwiderte Oliver nachdenklich; �denn der Himmel ist
    sehr fern, und die Seligen haben es dort zu gut, als da� sie an das
    Krankenbett eines armen Knaben herunterkommen sollten. Wenn sie es aber
    gewu�t hat, da� ich krank war, so hat sie gewi� Mitleid mit mir gehabt,
    denn sie war selbst sehr krank, ehe sie starb. Aber -- sie mag wohl
    nichts von mir wissen, denn wenn sie mich h�tte niederschlagen sehen,
    so w�rde sie sehr betr�bt geworden sein, und ihr Gesicht war immer so
    froh und vergn�gt, wenn ich von ihr getr�umt habe.�

    Frau Bedwin wischte sich die Augen, brachte ihm zu trinken und ermahnte
    ihn abermals, ganz still zu liegen, weil er sonst wieder krank werden
    w�rde. Er schwieg daher und hielt sich vollkommen ruhig, teils weil
    er der guten Frau nicht ungehorsam sein wollte, und andernteils, weil
    er durch das, was er gesagt hatte, bereits vollkommen ersch�pft war.
    Er schlief ein, und als er erwachte, stand ein Herr an seinem Bette,
    der seinen Puls f�hlte. �Nicht wahr, mein Kind, du f�hlst dich weit
    besser?� fragte ihn der Herr.

    �Ja, ich danke, Sir!� antwortete Oliver.

    �Das wu�te ich wohl. Und du bist hungrig -- nicht wahr?�

    �Nein, Sir.�

    �Hm! Ja, ganz recht. Du kannst auch in der Tat keinen Hunger empfinden.
    Er ist nicht hungrig, Frau Bedwin�, sagte der Herr mit sehr weiser
    Miene.

    Frau Bedwin neigte ehrfurchtsvoll den Kopf, wodurch sie andeuten zu
    wollen schien, da� sie den Doktor f�r einen �u�erst gescheiten Mann
    hielte. Der Doktor schien vollkommen derselben Meinung zu sein.

    �Du bist m�de, nicht wahr, mein Sohn?� sagte er.

    �Nein, Sir.�

    �Nicht?� wiederholte der Doktor; �das freut mich, und ich dachte es
    wohl. Aber durstig bist du?�

    �Ach ja, Sir�, erwiderte Oliver.

    �Ganz wie ich es erwartet habe. Frau Bedwin, es ist sehr nat�rlich, da�
    er Durst f�hlt. Sie k�nnen ihm ein wenig Tee mit Wei�brot ohne Butter
    geben. Halten Sie ihn nicht zu warm, Ma'am, und haben Sie acht, da� er
    nicht zu kalt wird.�

    Frau Bedwin knixte, und der Doktor ging. Oliver schlief bald wieder
    ein, und als er erwachte, war es fast zw�lf Uhr. Frau Bedwin sagte ihm
    gute Nacht und �berwies ihn der Pflege einer eingetretenen alten Frau,
    die in ihrem B�ndel ein kleines Gebetbuch und eine gro�e Nachtm�tze
    mitgebracht hatte, sich an den Kamin setzte und sehr bald einschlief.

    Oliver lag noch einige Zeit wach. Es herrschte eine feierliche Stille,
    und als er daran dachte, da� der Tod viele Tage und N�chte �ber seinem
    Bette geschwebt h�tte und das Gemach auch wohl noch mit Schmerz und
    Wehe erf�llen k�nnte, begann er inbr�nstig zu beten. Er versank darauf
    wieder in jenen festen Schlummer, den nur heitere Ruhe nach erduldeten
    Leiden gibt und aus welchem man nicht ohne Bedauern erwacht. Wenn es
    der Tod w�re -- wer m�chte aus ihm wieder aufwachen wollen zu den M�hen
    und �ngsten des Lebens, zu den N�ten der Gegenwart, den Sorgen um die
    Zukunft, und zumal den tr�ben Erinnerungen an die Vergangenheit!

    Es war heller Tag, als Oliver die Augen aufschlug, er f�hlte sich
    heiter und froh, die Krise war �berstanden, und er geh�rte der Welt
    wieder an. -- Nach drei Tagen konnte er, durch Kissen gest�tzt, in
    einem Lehnstuhle sitzen. Frau Bedwin lie� ihn in ihr kleines Zimmer
    hinunterbringen, setzte sich zu ihm an das Feuer und fing vor Freude
    von Herzen zu schluchzen an.

    �Sie sind sehr g�tig gegen mich, Ma'am�, sagte Oliver.

    Sie wollte nichts davon h�ren und bereitete ihm sorglich ein f�r seinen
    Zustand passendes Fr�hst�ck. Oliver heftete unterdes seine Blicke auf
    ein ihm gerade gegen�ber an der Wand h�ngendes Portr�t. Sie wurde
    aufmerksam darauf.

    �Magst du gern Bilder leiden, mein Kleiner?�

    �Ich habe noch wenige gesehen; aber wie sch�n und liebevoll das Gesicht
    der Dame ist!�

    �Ah, die Maler machen die Damen immer h�bscher, als sie sind,
    denn sie w�rden sonst keine Kundschaft haben. Der Mann, der die
    Konterfeimaschine erfand, h�tte vorauswissen k�nnen, da� es nichts
    damit w�re, denn es ist viel zu viel Ehrlichkeit dabei.�

    Sie lachte, Oliver aber blieb ernst und fragte: �Wen stellt denn das
    Bild vor, Ma'am?�

    �Ich wei� es nicht, mein Kind; aber sicher niemand, den wir beide
    kennen. Es scheint dir ja erstaunlich zu gefallen.�

    �Ach, es ist gar zu sch�n!� rief Oliver aus.

    �Du f�ngst doch nicht an, dich zu f�rchten?� sagte Frau Bedwin, denn
    sie gewahrte mit gro�er Verwunderung, da� Oliver das Portr�t mit einer
    Art von Beben betrachtete.

    �O, nein, nein,� erwiderte er rasch; �aber die Augen blicken so
    traurig, und es ist, als w�ren sie gerade, wo ich sitze, auf mich
    geheftet. Es macht mir das Herz schlagen�, setzte er mit leiser Stimme
    hinzu, �als wenn es lebte und zu mir reden wollte und k�nnte doch
    nicht.�

    �Gott sei uns gn�dig!� rief Frau Bedwin best�rzt aus; �sprich nicht
    so, Kind. Du mu�t noch sehr schwach und fieberisch sein. So, so -- nun
    kannst du es nicht mehr sehen.�

    Sie drehte bei diesen Worten seinen Stuhl herum; Oliver aber sah im
    Geiste das Bild so deutlich, als ob es ihm noch immer vor Augen hinge.
    Er wollte indes die gute alte Frau nicht �ngstigen und l�chelte ihr
    freundlich zu, als sie ihm seine Br�he mit Wei�brot brachte. Er hatte
    kaum einen L�ffel voll genossen, als Mr. Brownlow eintrat.

    Oliver sah noch sehr bla� und abgezehrt aus; er machte einen
    vergeblichen Versuch, aufzustehen, um seinem Wohlt�ter zu danken, dem
    die Tr�nen in die Augen traten.

    �Armes Kind, armes Kind�, sagte er. �Wie befindest du dich heute, mein
    Lieber?�

    �Vortrefflich, Sir�, erwiderte Oliver; �und ich bin Ihnen sehr dankbar
    f�r alle Ihre G�te.�

    �Gutes Kind,� sagte sein Wohlt�ter, erkundigte sich darauf, was ihm
    Frau Bedwin zur St�rkung gegeben, und bemerkte: �Br�he -- pfui! Ein
    paar Gl�ser Portwein w�rden ihm besser geschmeckt haben -- nicht wahr,
    Tom?�

    �Ich hei�e Oliver, Sir!� entgegnete der kleine Patient sehr verwundert.

    �Oliver! -- Wie? -- Oliver White?�

    �Nein, Sir, Twist -- Oliver Twist!�

    �Kurioser Name; -- warum sagtest du denn dem Richter, da� du White
    hie�est?�

    �Das hab' ich ihm ganz und gar nicht gesagt�, erwiderte Oliver �u�erst
    verwundert.

    Dies sah einer L�ge so �hnlich, da� ihn der alte Herr etwas strenge
    ansah. Allein es war unm�glich, seine Aussage zu bezweifeln, denn aus
    allen seinen Z�gen leuchtete die klarste Wahrheit hervor. Brownlow
    meinte, da� ein Mi�verst�ndnis obwalten m�sse, sein Verdacht schwand
    g�nzlich, und doch vermochte er die Blicke von Oliver nicht abzuwenden,
    denn abermals dr�ngte sich ihm die �hnlichkeit des Knaben mit bekannten
    Z�gen auf. Oliver hob flehend die Augen zu ihm empor.

    �Sie sind mir doch nicht b�se, Sir?�

    �Nein, nein; -- aber -- barmherziger Himmel! Was ist das? Frau Bedwin
    -- sehen Sie, sehen Sie!�

    Und w�hrend er hastig die Worte sprach, wies er nach dem Bilde �ber
    Olivers Lehnstuhl und dann auf Oliver selbst hin. Es konnte keine
    gr��ere �hnlichkeit geben; der Knabe war der Dame auf dem Bilde aus den
    Augen geschnitten.

    Oliver gewahrte die Ursache des pl�tzlichen Ausrufs seines Wohlt�ters
    nicht; der Schrecken war ihm zu viel gewesen; er war ohnm�chtig
    geworden. --

    Sobald der Baldowerer und Master Bates ihren Zweck erreicht hatten,
    alle Aufmerksamkeit von sich ab und auf Oliver zu lenken, schl�pften
    sie in eine Seitengasse, um eiligst nach Hause zur�ckzukehren. Sobald
    sie wieder zu Atem gekommen waren, fing Master Bates laut zu lachen
    an und rief sich und dem Freunde mit grenzenlosem Vergn�gen die
    unendlich spa�hafte Szene in das Ged�chtnis zur�ck, wie der ge�ngstete
    Oliver gelaufen und �berall angerannt war, und wie er selber und der
    Baldowerer ihn eifrigst mit gehetzt und das Tuch in der Tasche gehabt
    hatten. Sein Freund unterbrach jedoch bald seinen Redeflu� und warf das
    Bedenken auf, was Fagin dazu sagen w�rde?

    �Was soll er sagen?� meinte Charley.

    �Hm!� sagte Jack, pfiff und schnitt sehr bedeutsame Gesichter.

    Charley folgte ihm nachdenklich, bald darauf langten sie zu Hause an.
    Bei dem Ger�usch von Fu�tritten auf der krachenden Treppe fuhr der
    lustige alte Herr, der vor dem Feuer sa� und sich sein Mittagessen
    zubereitete, empor. Auf seinem wei�en Gesicht lag ein h�misches
    L�cheln, als er sich umdrehte und mit einem scharfen Blicke unter
    seinen dichten, roten Augenbrauen hervor sein Ohr der T�r zuwandte und
    horchte.

    �Wie? Was ist das?� murmelte der Jude erschrocken vor sich hin. �Nur
    zwei? Wo ist der dritte? Sie werden ihn in dem Gedr�nge doch nicht
    verloren haben? Horch!�

    Die Fu�tritte kamen n�her und n�her; endlich �ffnete sich die T�r, und
    der Baldowerer und Charley Bates traten in das Zimmer.




    13. Kapitel.

        Der Leser macht einige neue Bekanntschaften.


    �Wo ist Oliver?� fragte der Jude, sich drohend erhebend. �Wo ist der
    Junge?�

    Die jugendlichen Diebe sahen ihren Lehrmeister erschrocken �ber dessen
    Heftigkeit an und blickten unsicher einander an. Aber sie antworteten
    nicht.

    �Was ist aus dem Jungen geworden?� fragte der Jude, indem er den
    Baldowerer mit festem Griffe beim Kragen packte und f�rchterliche
    Verw�nschungen ausstie�. �Sprich, oder ich erdrossele dich! -- Willst
    du sprechen?� fuhr er fort, als keine Antwort erfolgte, und sch�ttelte
    den Baldowerer heftig.

    Charley erhob ein jammervolles Geheul, sein Freund ri� sich los,
    ergriff ein Messer und war im Begriff, es dem Juden in die Seite zu
    sto�en, als die T�r ge�ffnet wurde und ein Vierter, gefolgt von einem
    knurrenden, zerbissenen Hunde, eintrat.

    �Was gibt's hier, zu allen Teufeln? Spitzbube von Juden, was soll das
    bedeuten?�

    Die grobe, polternde Stimme geh�rte einem vierschr�tigen Manne von etwa
    f�nfundvierzig Jahren mit einem breiten Gesicht und d�ster grollendem
    Blicke an. Sein Bart war seit mehreren Tagen nicht abgenommen und das
    eine Auge von einem Schlage angeschwollen, den er erst vor kurzem
    erhalten haben mu�te. Arm- und Beinschellen dachte man sich bei der
    ganzen Erscheinung leicht hinzu.

    Er setzte sich gem�chlich. �Was sind das hier f�r Sachen?� fuhr er
    fort. �Warum mi�handelst du die Jungen, du alter, uners�ttlicher Filz
    und Pascher?[K] Ich wundere mich nur, da� sie dir die Kehle nicht
    abschneiden, was ich unfehlbar tun w�rde, wenn ich in ihrer Haut
    steckte. Ich h�tt's l�ngst getan, wenn ich dein Lehrling w�re. Freilich
    -- verkaufen h�tt' ich deinen Haut- und Knochenkadaver nicht k�nnen; du
    bist zu nichts gut, denn als ein merkw�rdiges St�ck von H��lichkeit in
    Spiritus aufbewahrt zu werden, und sie blasen so gro�e Gl�ser nicht.�

      [K] Hehler.

    �Pst, pst! Mr. Sikes,� fiel der zitternde Jude ein, �nicht so laut,
    nicht so laut!�

    �Ich will dich bemistern; du hast immer Teufeleien im Sinn, wenn du
    damit kommst. Du wei�t meinen Namen, und ich werd' ihm keine Unehre
    machen, wenn die Zeit kommt.�

    �Schon gut, schon gut; also Bill Sikes�, sagte der Jude kriechend
    dem�tig. �Ihr scheint �bler Laune zu sein, Bill.�

    Bill �berh�ufte ihn zur Erwiderung abermals mit Vorw�rfen und
    Schimpfw�rtern und deutete dabei auf so verd�chtige Dinge hin, da�
    ihn Fagin angstvoll und mit einem Seitenblicke nach den beiden Knaben
    fragte, ob er wahnsinnig geworden w�re. Bill machte pantomimisch einen
    Knoten unter seinem linken Ohre, wies durch eine Kopfbewegung �ber
    seine rechte Schulter, welche Symbolik der Jude vollkommen zu verstehen
    schien, forderte ein Glas Branntwein und f�gte die Erinnerung hinzu, es
    aber nicht zu vergiften. Er sagte dies scherzend; h�tte er jedoch den
    satanischen Blick sehen k�nnen, mit welchem der Jude sich umwendete, um
    nach dem Schranke zu gehen, so w�rde ihm die Warnung keineswegs unn�tig
    erschienen sein.

    Nachdem er einige Gl�ser hinuntergest�rzt, lie� er sich herab, die
    jungen Herren anzureden, was zu einem Gespr�ch f�hrte, in dessen Laufe
    ihm Olivers Gefangennehmung umst�ndlich und mit solchen Ausschm�ckungen
    erz�hlt wurde, wie sie der Baldowerer f�r n�tig erachtete.

    �Ich f�rchte, da� er wird etwas lehmern, wodurch wir kommen in
    Ungelegenheit�, bemerkte der Jude.

    �Sehr wahrscheinlich�, sagte Bill mit einem boshaften Grinsen. �Du bist
    verloren, Fagin.�

    Der Jude tat, als ob er die Unterbrechung nicht beachtet h�tte, behielt
    Sikes scharf im Auge und fuhr fort: �Ich f�rchte nur, wenn mir das
    Handwerk gelegt w�rde, m�cht's auch noch anderen mehr gelegt werden,
    und da� die Geschichte ein schlechteres Ende nimmt f�r Euch, als f�r
    mich, mein Lieber.�

    Sikes fuhr zusammen und blickte den Juden w�tend an, der jedoch
    die Achseln zuckend gerade vor sich hinstarrte. Nach einem langen
    Stillschweigen sagte er mit leiserer Stimme: �Wir m�ssen zu erfahren
    suchen, was sich auf der Polizei zugetragen hat.�

    Fagin nickte beif�llig.

    �Hat er nichts ausgeschwatzt und ist ein Haftbefehl gegen ihn
    ausgestellt worden, so ist nichts zu bef�rchten, bis er wieder
    loskommt; dann aber m�ssen wir seiner so bald wie m�glich wieder
    habhaft zu werden suchen.�

    Der Jude nickte abermals. Der Rat war offenbar gut, nur war die
    Ausf�hrung schwierig, da alle vier Gentlemen einen un�berwindlichen
    Widerwillen dagegen hegten, einem Polizeiamte nahezukommen. Sie
    blickten einander verlegen an, als die beiden jungen Damen eintraten,
    deren Bekanntschaft Oliver vor einigen Tagen gemacht hatte. Der Fall
    wurde ihnen vorgetragen, und Fagin sprach seine Zuversicht aus, da�
    Betsy den Auftrag �bernehmen werde. Die junge Dame war zu wohlerzogen
    und zu feinf�hlend, um einem Mitgliede der Gesellschaft geradezu
    oder vielleicht gar mit Sch�rfe zu widersprechen oder eine Bitte
    abzuschlagen. Sie sagte daher keineswegs entschieden nein, sondern
    begn�gte sich mit der Versicherung, da� sie sich h�ngen lassen wollte,
    wenn sie's t�te.

    Der Jude wendete sich an ihre Freundin: �Liebe Nancy, was sagst du?�

    �Da� ich mich sch�nstens h�ten werde; also gebt Euch nur weiter keine
    M�he, Fagin.�

    �Wie soll ich das nehmen?� fiel Sikes grollend ein.

    �Just wie ich's gesagt habe, Bill�, entgegnete die Dame sehr ruhig.

    �Du bist aber eben die rechte Person dazu; es kennt dich hier herum
    niemand.�

    �Und es tut auch gar nicht not, da� mich jemand kennen lernt, was ganz
    gegen meinen Wunsch w�re.�

    �Sie geht, Fagin�, sagte Sikes.

    �Nein, sie l��t's wohl bleiben�, eiferte Nancy.

    �Ja, ja, sie geht doch�, wiederholte Sikes.

    Und er hatte recht. Nancy lie� sich endlich durch Geschenke,
    Versprechungen und Drohungen bewegen, den Auftrag zu �bernehmen. Auch
    hatte sie in der Tat weniger als ihre Freundin zu besorgen, mit einem
    ihrer zahlreichen Bekannten zusammenzutreffen, da sie erst seit ganz
    kurzer Zeit die entlegene, sehr anst�ndige Vorstadt Ratcliffe mit
    der Gegend von Fieldlane vertauscht hatte. Der Jude staffierte sie
    aus seinen unersch�pflichen Vorr�ten so aus, wie es dem Zwecke am
    angemessensten erschien, und gab ihr einen Korb und einen Hausschl�ssel
    in die Hand.

    �Ach, mein Bruder! mein armer, lieber, kleiner Bruder�, begann Nancy
    mit �berstr�menden Tr�nen und h�nderingend zu wehklagen. �Ach, was
    ist aus meinem Bruder geworden -- wo soll ich ihn finden? O haben Sie
    Erbarmen, liebe Herren, und sagen Sie mir, was aus ihm geworden ist!�

    Ihre Zuh�rer waren entz�ckt; sie hielt inne, blinzelte l�chelnd und
    bedeutungsvoll und verschwand.

    �Die Nancy ist 'ne gescheite Dirne�, sagte der Jude mit feierlichem,
    nachdenklichem Kopfnicken zu seinen beiden jungen Freunden, als wenn er
    sie mahnen wollte, das eben geschaute gl�nzende Beispiel nachzuahmen.

    �Sie ist 'ne Zierde ihres Geschlechts�, stimmte Sikes, sein Glas
    f�llend und nachdr�cklich auf den Tisch schlagend, ein. �Sie lebe hoch,
    und m�chten ihr alle gleich werden!�

    Die Vielgepriesene eilte unterdes nach dem Polizeiamte, wo sie bald,
    trotz ein wenig nat�rlicher Sch�chternheit, allein und ohne Besch�tzer
    die Stra�en zu durchwandern, gl�cklich und ohne Gef�hrde anlangte. Nach
    einigen mi�lungenen Versuchen wendete sie sich weinend und wehklagend
    an den Gef�ngnisw�rter, von welchem sie in Erfahrung brachte, da�
    Olivers Unschuld ans Licht gekommen und da� er von dem beraubten Herrn
    mit fortgenommen worden sei, der in der Gegend von Pentonville wohne,
    wohin zu fahren er den Kutscher angewiesen habe. Mit dieser Auskunft
    kehrte sie zum Juden zur�ck.

    Sobald sie ihren Bericht erstattet hatte, rief Bill Sikes hastig seinen
    Hund, st�lpte den Hut auf den Kopf und entfernte sich, ohne sich Zeit
    zu der Formalit�t zu nehmen, der Gesellschaft einen guten Morgen zu
    w�nschen.

    �Wir m�ssen ihn ausfindig machen; wir m�ssen wissen, wo er steckt�,
    sagte der Jude in gro�er Aufregung. �Charley, geh auf die Lauer, bis du
    etwas von ihm siehst oder h�rst. Beste Nancy, ich mu� ihn wiederhaben
    -- ich verlasse mich ganz auf dich und den Baldowerer. Da, da habt ihr
    Geld. Ich entferne mich heut' abend von hier -- ihr wi�t, wo ich zu
    finden bin. Macht, da� ihr fortkommt -- ihr d�rft keinen Augenblick
    l�nger hierbleiben.�

    Er stie� alle hinaus, verschlo� die T�r hinter ihnen und steckte seine
    Kostbarkeiten zu sich. �Er hat nichts ausgeschwatzt auf der Polizei�,
    murmelte er; �tut er's aber gegen die Leute, bei denen er sich jetzt
    aufh�lt -- wir werden ihn wiederbekommen und wollen ihm schon stopfen
    den Mund.�




    14. Kapitel.

        In welchem Mr. Grimwig auftritt.


    Oliver erholte sich bald wieder von der Ohnmacht, in die er bei dem
    kurzen Ausrufe Mr. Brownlows gefallen war. Der alte Herr und Frau
    Bedwin vermieden sorgf�ltig jedes Gespr�ch, durch das er wieder an
    das Bild oder seine Herkunft und Lage h�tte erinnert werden k�nnen,
    und suchten ihn auf jede Weise angenehm zu unterhalten, ohne ihn
    aufzuregen. Als er jedoch am folgenden Tage wieder in das Zimmer der
    Haush�lterin herunterkam, hob er sogleich die Augen nach der Wand
    empor, in der Hoffnung, das Bild der sch�nen Dame zu erblicken. Er sah
    sich get�uscht; es war entfernt worden. Frau Bedwin hatte ihn jedoch
    beobachtet.

    �Ah!� sagte sie, �es ist nicht mehr da, mein Kind.�

    �Ich seh' es, Ma'am!� erwiderte Oliver seufzend. �Warum ist es denn
    fortgenommen worden?�

    �Weil Mr. Brownlow sagte, es schiene dich unruhig zu machen und k�nnte
    daher deiner Wiederherstellung schaden.�

    �Ach, es machte mich gar nicht unruhig, Ma'am. Ich freute mich, es
    anzusehen, und hatte es gar zu lieb gewonnen.�

    �Nun, nun, mein Kind,� sagte die gute Frau, �es geht dir ja zusehends
    besser, und es soll schon wieder aufgeh�ngt werden; ich verspreche es
    dir. La� uns jetzt aber von anderen Dingen sprechen.�

    Sie hatte ihm in seiner Krankheit so viel Liebe erwiesen, da� er
    sich vornahm, einstweilen nicht mehr an das Bild zu denken. Er h�rte
    ihr daher aufmerksam zu, als sie begann, ihm von ihren wohlgeratenen
    Kindern und ihrem guten, seligen Ehemann zu erz�hlen. Sodann wurde
    Tee getrunken, worauf sie ihn Cribbage spielen lehrte, was er schnell
    begriff und eifrig mit ihr spielte, bis es Zeit war, zu Bett zu gehen.

    Es folgten nun selige Tage f�r Oliver. Alles um ihn her war so still,
    sauber und ordentlich, und jedermann war so liebevoll gegen ihn, da�
    er fast im Himmel zu sein glaubte. Als er imstande war, sich wieder
    ordentlich anzukleiden, hatte Mr. Brownlow schon f�r einen ganz neuen
    Anzug gesorgt, und da ihm gesagt wurde, er k�nnte mit seinen alten
    Kleidern tun, was er wollte, so gab er sie der Magd, die sehr gef�llig
    gegen ihn gewesen war, und sagte ihr, sie m�chte sie an einen Juden
    verkaufen und das Geld behalten. Die Magd machte sogleich Gebrauch von
    der erhaltenen Erlaubnis, Oliver sah durch das Fenster, wie der Jude
    seine ganze alte Garderobe zusammenwickelte, einsackte und fortging;
    und er freute sich nicht wenig dar�ber, da er nun nicht mehr zu
    f�rchten brauchte, die traurigen Lumpen je wieder anlegen zu m�ssen.

    Es mochte etwa eine Woche vergangen sein, als eines Nachmittags Mr.
    Brownlow herunterschickte und Oliver zu sich rufen lie�. Frau Bedwin
    ordnete eiligst den Anzug und das Haar ihres kleinen Pfleglings und
    begleitete ihn selbst bis an Mr. Brownlows T�r. Das Zimmer war mit
    B�chern angef�llt, und das einzige Fenster wies in einen kleinen
    Blumengarten. Mr. Brownlow legte ein Buch aus der Hand und sagte
    Oliver, er m�chte n�her kommen und sich setzen. Oliver tat, wie
    ihm gehei�en war, und dachte, wo die Leute wohl gefunden werden
    k�nnten, eine solche Menge von B�chern zu lesen, die geschrieben zu
    sein schienen, um die Welt kl�ger zu machen -- eine Sache, welche
    fortw�hrend erfahreneren Leuten zu schaffen macht, als Oliver Twist es
    war.

    �Du siehst hier sehr viel B�cher, nicht wahr, mein Kind?� fragte Mr.
    Brownlow.

    �Ja, sehr viele�, erwiderte Oliver; �ich habe noch nie eine solche
    Menge von B�chern gesehen.�

    �Du sollst sie, wenn du dich gut betr�gst, auch lesen, was dir noch
    besser gefallen wird als das blo�e Beschauen der B�nde -- wenn auch
    nicht immer; denn es gibt allerdings B�cher, an welchen die Einb�nde
    bisweilen das Beste sind. M�chtest du wohl ein recht gescheiter Mann
    werden und selbst B�cher schreiben?�

    �Ich m�chte lieber in B�chern lesen, Sir�, entgegnete Oliver.

    �Wie, du m�chtest also kein B�cherschreiber sein?� sagte der alte Herr.

    Oliver besann sich ein wenig und erwiderte endlich, es bed�nke ihn weit
    besser, ein Buchh�ndler zu sein, wor�ber der alte Herr herzlich lachte,
    und wozu er bemerkte, Oliver habe da etwas sehr Gescheites gesagt.
    Oliver freute sich �ber diese Anerkennung, obgleich er durchaus nicht
    begriff, wodurch er sie verdient haben m�chte.

    �Sei nur ohne Furcht�, sagte der alte Herr; �ich werde dich nicht zum
    Schriftsteller machen, solange es noch ein anderes ehrliches Gesch�ft
    oder Handwerk gibt, das du erlernen kannst.�

    �Ich danke, Sir�, entgegnete Oliver, und der alte Herr lachte abermals
    �ber den gro�en Ernst, mit dem er antwortete, und sagte ein paar Worte
    von einem merkw�rdigen Instinkt, welche Oliver nicht sehr beachtete,
    da er sie nicht verstand. Brownlow fuhr darauf in einem wom�glich noch
    freundlicheren, aber zugleich ernsteren Tone, als er gegen Oliver bis
    dahin angenommen, fort: �Sei jetzt recht aufmerksam auf das, was ich
    dir sagen werde. Ich denke ohne R�ckhalt mit dir zu reden, weil ich
    �berzeugt bin, da� du mich ebensogut verstehen wirst wie viel �ltere
    Personen.�

    Oliver erschrak. �Ach!� rief er aus, �sagen Sie nicht, da� Sie mich
    fortschicken wollen, Sir; weisen Sie mir nicht die T�r, da� ich wieder
    auf den Stra�en umherirren mu�. Lassen Sie mich bei Ihnen bleiben und
    Ihnen dienen. Schicken Sie mich nicht in das schreckliche Haus zur�ck,
    woher ich gekommen bin. Erbarmen Sie sich eines armen, verlassenen
    Knaben, bester Herr!�

    �Mein liebes Kind,� sagte der alte Herr ger�hrt, �du brauchst nicht
    zu f�rchten, da� ich meine Hand von dir abziehe, solange du mir keine
    Ursache dazu gibst.�

    �Das will ich nie, niemals, Sir!�

    �Ich hoffe, da� du es nicht tun wirst, glaube es auch nicht. Ich bin
    oft get�uscht und betrogen von Leuten, denen ich wohltun wollte, bin
    aber trotzdem sehr geneigt, dir zu vertrauen, und ich empfinde eine
    gr��ere Teilnahme f�r dich, als ich sie sogar mir selbst erkl�ren kann.
    Die ich am meisten geliebt habe, ruhen l�ngst in ihren Gr�bern, und ich
    habe auch meines Lebens Gl�ck und Zier begraben -- nicht aber meine
    Herzensw�rme. Auch herber Kummer hat sie nicht ausgel�scht, sondern
    nur noch st�rker angefacht; wie denn allerdings Schmerz und Leid unser
    Inneres stets reinigen und l�utern sollten.� -- Er hatte dies mit
    leiser Stimme und mehr vor sich hin als zu Oliver gesprochen, der ganz
    still dasa� und kaum zu atmen wagte. -- �Doch ich sagte das nur,� fuhr
    der alte Herr wieder heiterer fort, �weil du ein junges Gem�t hast,
    und wenn du wei�t, da� ich viel gelitten habe, dich vielleicht noch
    sorgf�ltiger h�ten wirst, mir abermals wehe zu tun. Du sagst, da� du
    eine Waise w�rest und ganz allein in der Welt dast�ndest. Alles, was
    ich in Erfahrung habe bringen k�nnen, best�tigt deine Angaben. Erz�hle
    mir nun, wer deine Eltern gewesen sind, wo du erzogen und wie du in die
    Gesellschaft geraten bist, in welcher ich dich gefunden habe. Sage die
    Wahrheit, und wenn ich finde, da� du kein Verbrechen begangen hast, so
    soll es dir niemals, solange ich lebe, an einem Freunde fehlen.�

    Oliver vermochte vor Schluchzen ein paar Minuten nicht zu antworten,
    und als er sich endlich gefa�t hatte und seine Erz�hlung beginnen
    wollte, lie� sich ein Herr zum Tee anmelden.

    �Es ist ein Freund von mir, Mr. Grimwig�, sagte Brownlow zu Oliver.
    �Er hat ein wenig rauhe Manieren, ist aber im Herzen ein sehr wackerer
    Mann.�

    Oliver fragte, ob er hinuntergehen solle, allein Brownlow hie� ihn
    bleiben, und in demselben Augenblick trat Mr. Grimwig, ein korpulenter
    alter Herr, gest�tzt auf einen t�chtigen Handstock -- denn er hatte
    ein etwas lahmes Bein --, schon in das Zimmer. Oliver hatte nie ein
    so verzwicktes Gesicht gesehen. Grimwig hielt dem Freunde sogleich
    auf Armesl�nge ein St�ckchen Zitronenschale entgegen und polterte,
    dergleichen w�rde ihm �berall in den Weg geworfen. �Ich will meinen
    eigenen Kopf aufessen, wenn Zitronenschale nicht noch mein Tod ist!�
    beteuerte er.

    Es war seine gew�hnliche Beteuerung; allein wenn die Erfindung, den
    eigenen Kopf zu verspeisen, auch noch gemacht werden sollte, so w�rde
    es einem Herrn, wie Mr. Grimwig war, doch jedenfalls stets sehr
    schwerfallen, in einer einzigen Mahlzeit damit zustande zu kommen.

    Mr. Grimwig erblickte Oliver, trat ein paar Schritte zur�ck und fragte
    Brownlow verwundert, wer der Knabe w�re.

    �Der Oliver Twist, von welchem ich Ihnen erz�hlt habe�, erwiderte
    Brownlow.

    Oliver verbeugte sich.

    �Doch nicht der Knabe, der das Fieber gehabt hat?� sagte Grimwig, sich
    noch etwas weiter zur�ckziehend.

    �Gehabt hat�, wiederholte Brownlow l�chelnd.

    Grimwig setzte sich, ohne seinen Handstock zur Seite zu stellen,
    be�ugelte den hocherr�tenden Oliver durch seine Lorgnette und redete
    ihn nach einiger Zeit an. �Wie befindest du dich?�

    �Danke, Sir, sehr viel besser�, erwiderte Oliver.

    Brownlow schien zu besorgen, da� sein absonderlicher Freund etwas
    Unangenehmes sagen m�chte, und hie� Oliver daher hinuntergehen und Frau
    Bedwin ank�ndigen, da� die Herren den Tee erwarteten. Oliver ging mit
    Freuden.

    �Er ist ein artig aussehender Knabe, nicht wahr?�

    �Kann's nicht sagen�, entgegnete Grimwig verdrie�lich.

    �Sie k�nnen es nicht sagen?�

    �Nein. Ich kann nie einen Unterschied an Knaben entdecken. Ich
    kenne nur zwei Arten von Knaben -- Milchsuppengesichter und
    Rindfleischgesichter.�

    �Zu welcher Art geh�rt Oliver?�

    �Zu den Milchsuppengesichtern. Ich kenne einen Freund, der einen Knaben
    mit einem Rindfleischgesicht hat -- einen sch�nen Knaben, wie ihn seine
    Eltern nennen, mit rundem Kopf, roten Wangen und gl�nzenden Augen --
    einen abscheulichen Knaben, wie ich ihn nenne -- mit einem K�rper und
    Gliedern, die die N�hte seines blauen Anzugs zu sprengen drohen, mit
    der Stimme eines Matrosen und einem Wolfshunger. Ich kenne ihn -- den
    Bengel!�

    �Dann gleicht er Oliver nicht, dem Sie daher nicht z�rnen d�rfen.�

    �Freilich gleicht er dem Oliver nicht, der vielleicht noch schlimmer
    ist.�

    Brownlow hustete ungeduldig, was seinen Freund h�chlich zu erg�tzen
    schien.

    �Ja, ja, er ist vielleicht noch schlimmer�, fuhr Grimwig fort. �Woher
    stammt er? Wer ist er? Was ist er? Er hat ein Fieber gehabt. Gute
    Menschen pflegen keine Fieber zu bekommen, wohl aber schlechte. Ich
    habe einen Menschen gekannt, der in Jamaika aufgeh�ngt wurde, weil er
    seinen Herrn ermordet hatte. Er hatte sechsmal das Fieber gehabt und
    wurde deshalb nicht zur Begnadigung empfohlen.�

    Grimwig war im innersten Herzensgrunde sehr geneigt, anzuerkennen, da�
    Oliver ein au�erordentlich einnehmender Knabe w�re; allein er liebte
    noch mehr den Widerspruch, die Zitronenschale hatte ihn gereizt,
    er war entschlossen, sich von niemand sein Urteil �ber einen Knaben
    vorschreiben zu lassen, und hatte sich aus diesen triftigen Gr�nden
    von Anfang an vorgenommen, seinem Freunde in allem zu widersprechen.
    Als Brownlow daher zugestand, da� seine bisherigen Erkundigungen noch
    ungen�gend w�ren, l�chelte Grimwig ziemlich boshaft und fragte, ob
    die Haush�lterin auch wohl regelm��ig das Silbergeschirr nachs�he und
    wegschl�sse, denn er w�rde sich eben nicht wundern, wenn sie einmal
    einige L�ffel oder dergleichen vermi�te, usw.

    Brownlow, obgleich selbst etwas heftigen Temperaments, ertrug dies
    alles sehr gutlaunig, da er die Sonderbarkeiten seines Freundes kannte;
    und da sich dieser mit dem Tee und den Semmeln zufrieden zeigte, so
    ging alles weit besser, als man h�tte erwarten sollen, und Oliver, der
    wieder heraufgerufen war, f�hlte sich in des sauert�pfischen Herrn
    Anwesenheit leichter als zuvor. Als das Teegeschirr hinwegger�umt
    wurde, fragte Grimwig, wann sein Freund den Knaben zu veranlassen
    ged�chte, ihm einen ausf�hrlichen und wahrhaften Bericht �ber seine
    Lebensumst�nde und Schicksale zu erstatten?

    �Morgen fr�h�, erwiderte Brownlow. �Ich w�nsche dabei unter vier Augen
    mit ihm zu sein. Komm morgen vormittag um zehn Uhr zu mir herauf,
    Oliver.�

    �Ja, Sir�, sagte Oliver. Er antwortete mit einigem Stocken, weil er
    dadurch in Verwirrung geraten war, da� Mr. Grimwig ihn bei seiner Frage
    so scharf angesehen hatte.

    �Ich will Ihnen etwas sagen�, fl�sterte Grimwig Brownlow in das Ohr;
    �er kommt morgen fr�h nicht herauf zu Ihnen. Ich habe ihn beobachtet.
    Er betr�gt Sie, lieber Freund.�

    �Ich schw�re darauf, da� er's nicht tut�, entgegnete Brownlow mit W�rme.

    �Ich will meinen Kopf aufessen, wenn er's nicht tut.�

    �Und ich b�rge mit meinem Leben f�r seine Wahrhaftigkeit.�

    �Und ich mit meinem Kopfe f�r seine L�genhaftigkeit.�

    �Wir werden sehen�, sagte Brownlow, seinen Unwillen bemeisternd.

    �Ja, ja, wir werden allerdings sehen�, wiederholte Grimwig mit einem
    herausfordernden L�cheln.

    Das Schicksal wollte es, da� gerade in diesem Augenblick Frau Bedwin
    mit einigen B�chern hereintrat, welche Brownlow an demselben Tage von
    dem mehrerw�hnten Buchh�ndler gekauft hatte. Sie legte sie auf den
    Tisch und schickte sich an, wieder hinauszugehen.

    �Lassen Sie den Ladenburschen noch warten�, sagte Brownlow; �er soll
    etwas mit zur�cknehmen -- ein P�ckchen B�cher und das Geld f�r die
    gekauften.�

    Der Ladenbursche war aber schon wieder fortgegangen.

    �Ah, das ist mir aber sehr unangenehm�, fuhr Brownlow fort. �Der Mann
    braucht sein Geld, und ich w�rde es auch gern gesehen haben, da� er die
    B�cher noch heute zur�ckerhalten h�tte.�

    �Schicken Sie sie doch durch Oliver�, fiel Grimwig mit einem ironischen
    L�cheln ein. �Sie wissen, er wird sie ohne Zweifel richtig abliefern.�

    �Ja, lassen Sie sie mich hintragen, Sir�, sagte Oliver eifrig. �Ich
    will auch den ganzen Weg laufen.�

    Brownlow wollte eben erkl�ren, da� er Oliver unter keiner Bedingung
    hinschicken werde, als ein boshaftes Husten seines Freundes ihn
    bestimmte, seinen Beschlu� abzu�ndern, um Grimwig der Ungerechtigkeit
    seines Argwohns zu �berf�hren. Er hie� Oliver die B�cher hintragen
    und gab ihm zugleich eine F�nfpfundnote, worauf er zehn Schillinge
    zur�ckbekommen w�rde.

    Oliver versicherte, er w�rde in zehn Minuten wieder da sein, verbeugte
    sich ehrerbietig und eilte hinaus. Frau Bedwin folgte ihm vor die
    Haust�r, gab ihm ausf�hrliche Anweisungen in betreff des n�chsten Weges
    und entlie� ihn unter vielen wiederholten Ermahnungen, sich nicht zu
    �berlaufen, sich nicht zu erk�lten usf. Es war ihr h�chst unangenehm,
    ihn aus den Augen lassen zu m�ssen. Sie h�tte auf Mr. Brownlow z�rnen
    m�gen und sah Oliver nach, bis er an der n�chsten Ecke angelangt war,
    wo er sich noch einmal umwandte und ihr freundlich zunickte.

    �Er ist in h�chstens zehn Minuten wieder hier�, sagte Brownlow und
    legte seine Uhr auf den Tisch. �Es wird bis dahin dunkel geworden sein.�

    �Sie glauben also wirklich, da� er wiederkommt?�

    �Sie nicht?� entgegnete Brownlow l�chelnd.

    In seinem Freunde regte sich der Widerspruchsgeist gerade mit
    besonderer Lebhaftigkeit, und Brownlows L�cheln verst�rkte ihn noch.
    �Nein!� erwiderte er mit gro�er Bestimmtheit. �Er steckt in einem
    nagelneuen Anzuge, hat ein Paket wertvoller B�cher unter dem Arme und
    eine F�nfpfundnote in der Tasche; er wird sich sofort wieder zu seinen
    alten Spie�gesellen begeben und Sie auslachen. Ich will meinen Kopf
    aufessen, wenn er sich jemals wieder hier blicken l��t.�

    Er r�ckte n�her an den Tisch, und beide sa�en in stummer Erwartung da.
    Es ist der Bemerkung wert und wirft ein Licht auf die Bedeutung, welche
    wir unseren eigenen Urteilen beilegen, und den Stolz, mit welchem wir
    uns auf unsere �bereiltesten Schl�sse verlassen, da� Grimwig, obgleich
    er kein schlechtes Herz hatte, obgleich es ihn wirklich betr�bt
    haben w�rde, wenn er seinen gesch�tzten Freund betrogen gesehen, im
    Augenblick ebenso lebhaft w�nschte wie hoffte, Oliver m�chte nicht
    wiederkommen. Aus solchen Widerspr�chen ist die menschliche Natur
    zusammengesetzt!

    Es wurde so dunkel, da� die Zahlen auf dem Zifferblatt der Uhr
    nicht mehr zu erkennen waren; allein die beiden alten Herren sa�en
    fortw�hrend da und hefteten schweigend die Blicke auf die Uhr.




    15. Kapitel.

        Was Oliver auf dem Wege zum Buchh�ndler begegnete.


    Olivers R�ckkehr wurde beiden Herren immer zweifelhafter, zu Grimwigs
    Triumph und Brownlows tiefer Betr�bnis. Ich h�tte nun hier in meinem
    Prosaepos die kostbarste Veranlassung, die Leser mit vielen weisen
    Betrachtungen �ber die offenbare Unklugheit zu unterhalten, seinen
    Mitmenschen Gutes zu erweisen ohne Aussicht auf irdischen Lohn,
    oder vielmehr dar�ber, wie sehr es die Klugheit erfordere, in einem
    besonders hoffnungslosen Falle einige Liebe und Menschenfreundlichkeit
    an den Tag zu legen, und sodann dergleichen Schwachheiten f�r immer
    abzulegen. Die Vorteile liegen auf der Hand. H�lt sich der, dem ihr
    unter die Arme gegriffen, gut und dient ihm euer geleisteter Beistand
    zum Wohlergehen, so erhebt er euch bis in den Himmel, ihr werdet
    sehr geachtete Leute und gelangt in den Ruf, unendlich viel Gutes im
    Verborgenen zu tun, wovon nur der zwanzigste Teil bekannt werde; zeigt
    er sich als ein Undankbarer und Nichtsw�rdiger, so habt ihr euch in die
    vortreffliche Stellung gebracht, da� man euch nachsagt, ihr h�ttet euch
    h�chst uneigenn�tzig, mildt�tig und dienstfertig erwiesen, w�ret nur
    durch erfahrenen Undank und Verrat menschenfeindlich geworden, und man
    k�nne euch euer Gel�bde nicht verdenken, nie wieder einem Menschenkinde
    beizuspringen, um nicht durch abermalige T�uschungen verletzt
    zu werden. Ich kenne eine Menge Personen, welche die angegebene
    Klugheitsregel befolgt haben, und kann versichern, da� sie in der
    allgemeinsten und nat�rlich verdientesten Achtung stehen.

    Brownlow geh�rte indes zu ihrer Zahl nicht, denn er blieb hartn�ckig
    dabei, Gutes zu tun um des Guten selbst und um der Herzensberuhigung
    und Freude willen, die es ihm gew�hrte. T�uschungen raubten ihm sein
    Vertrauen und seine Milde und seine Menschenfreundlichkeit nicht, und
    Undankbarkeit von seiten einzelner f�hrte ihn nicht zu dem Entschlusse,
    sich daf�r an der ganzen leidenden Menschheit zu r�chen. Ich werde
    daher die fraglichen vielen weisen Betrachtungen unangestellt lassen,
    und sollte dieser Grund ungen�gend erscheinen, so kann ich noch
    hinzuf�gen, da� es obendrein g�nzlich au�er meiner urspr�nglichen
    Absicht liegt.

    Im finsteren Gastzimmer einer kl�glichen Winkelschenke, gelegen in der
    schmutzigsten Gasse von Little Saffron Hill, sa� bei einem Bierkruge
    und Branntweinglase ein Mann, in welchem trotz des herrschenden
    Halbdunkels kein irgend erfahrener Polizeiagent Bill Sikes verkannt
    haben w�rde. Zu seinen F��en lag sein wei�er, rot�ugiger Hund, und
    sei es, da� Bill seine Zeit nicht besser anzuwenden wu�te, oder da�
    er seine �ble Laune an irgendeinem Gegenstande auszulassen w�nschte,
    genug, er versetzte dem Tiere einen derben Fu�tritt. Dem Hunde mi�fiel
    der offenbare Mutwille dieser Behandlung so sehr, da� er nach seines
    Herrn Beinen schnappte, Bill ergriff w�tend das Sch�reisen und sein
    Messer, als die T�r sich auftat und der Hund hinausscho�. Zu einem
    Streite geh�ren dem Sprichworte gem�� zwei, und Bill setzte daher den
    einmal begonnenen sogleich mit dem Eintretenden fort.

    �Verdammter Jude, was trittst du zwischen mich und meinen Hund?� schrie
    er ihm entgegen.

    �Ich wu�t's ja nicht, mein Lieber, wu�t's ja nicht, da� Ihr wolltet dem
    Hunde zu Leibe�, erwiderte Fagin dem�tig.

    �Spitzbube, hast du den L�rm nicht geh�rt?�

    �So wahr mir Gott gn�dig ist, nein, Bill, nicht 'nen einzigen Laut.�

    �Ja freilich, du h�rst nichts, gar nichts�, entgegnete Sikes h�hnisch;
    �ebenso wie du selbst ein und aus schleichst, ohne da� man dich h�rt.
    Ich wollte nur, da� du jetzt der Hund w�rst.�

    �Warum denn?� fragte Fagin mit einem gezwungenen L�cheln.

    �Weil die Regierung, die das Leben solcher Halunken sch�tzt, wie du
    einer bist, und die nicht halb so viel Mut haben wie die schlechtesten
    Hunde, jedermann erlaubt, seinen Hund abzuschlachten, wenn's ihm
    beliebt -- darum!� erwiderte Sikes, sein Messer mit einem sehr
    bedeutungsvollen Blicke wieder einsteckend.

    Der Jude rieb sich die H�nde, setzte sich an den Tisch und zwang sich,
    �ber die Spa�haftigkeit seines Freundes zu lachen, jedoch war ihm
    offenbar dabei nicht besonders wohl zumute.

    �Grinse nur, ja grinse nur�, sagte Sikes, ihn mit ver�chtlichem Trotze
    anblickend; ��ber mich sollst du doch nicht lachen, es m��te denn unter
    der Nachtm�tze sein am Galgen. Ich habe die Hand oben, Fagin, und will
    verdammt sein, wenn ich dir den Daumen nicht auf'm Auge halte. Baumele
    ich, baumelst du auch; also h�te dich vor mir und trag' h�bsch Sorge
    f�r mich.�

    �Schon gut, mein Lieber�, fiel der Jude ein; �ich wei� das alles;
    Gewinn und Gefahr ist gemeinschaftlich bei uns.�

    �Hm!� murrte Sikes, als wenn er d�chte, der Gewinn m�chte wohl zumeist
    auf des Juden Seite sein. �Was hast du mir denn aber zu sagen?�

    �'s ist alles in den Schmelztiegel gewandert und gl�cklich wieder
    heraus -- da ist Euer Anteil. Ihr erhaltet eigentlich mehr, als Ihr
    solltet, mein Lieber; doch da ich wei�, da� Ihr mir schon mal wieder
    sein werdet gef�llig, und --�

    �Haltet ein mit dem Schw�tzen�, unterbrach ihn Sikes ungeduldig. �Wo
    ist's? Her damit!�

    �Ja, ja doch, Bill; g�nnt mir nur Zeit. Da ist's�, versetzte Fagin,
    zog ein altes, baumwollenes Taschentuch hervor, kn�pfte einen Knoten
    auf und reichte Sikes ein P�ckchen, der es �ffnete und die Goldst�cke
    hastig zu z�hlen anfing.

    �Ist das alles?� fragte Sikes.

    �Ja, alles.�

    �Hast du auch das P�ckchen nicht aufgemacht auf dem Wege und ein paar
    St�ck verschluckt? Stell dich nur nicht beleidigt -- hast's ja schon
    oft getan. Greif an den Bimbam.�

    Fagin klingelte, und es erschien ein anderer Jude, der j�nger war, aber
    nicht weniger absto�end und spitzb�bisch aussah. Sikes wies stumm nach
    dem leeren Kruge hin. Jener verstand den Wink und ging wieder hinaus,
    jedoch nicht, ohne Fagin vorher einen Blick zugeworfen zu haben, den
    dieser durch ein kaum bemerkbares Kopfsch�tteln beantwortete. Sikes
    hatte sich zuf�llig geb�ckt; h�tte er den Blick des einen und das
    Kopfsch�tteln des anderen Juden gewahrt, so m�chte er der Meinung
    gewesen sein, da� ihm diese Pantomimen nichts Gutes bedeuteten.

    �Ist niemand hier, Barney?� fragte Fagin den wieder eintretenden Juden.

    �Blo� Mi� Nancy.�

    �Schick sie herein!� sagte Sikes.

    Barney blickte Fagin fragend an, ging und kehrte gleich darauf mit
    Nancy zur�ck.

    �Du bist auf der Spur, Nancy, nicht wahr, mein Engel?� fragte Bill und
    reichte ihr ein gef�lltes Glas.

    �Ja, Bill�, erwiderte die junge Dame, nachdem sie das Glas geleert
    hatte; �hab' aber M�he genug gehabt. Er ist krank gewesen und --�

    Nancy bemerkte ein Augenzwinkern Fagins, das eine Warnung vor
    �bergro�er Mitteilsamkeit zu bedeuten schien. Sie brach ab und fing
    an von anderen Gegenst�nden zu reden. Nach zehn Minuten bekam Fagin
    einen Husten, worauf Nancy erkl�rte, da� es Zeit sei, zu gehen. Sikes
    sagte, da� er sie eine Strecke begleiten wolle, da er denselben Weg
    habe. Sie entfernten sich daher miteinander. Der Hund folgte in einiger
    Entfernung. Fagin sah Sikes durch das Fenster nach, sch�ttelte die
    geballte Faust hinter ihm, murmelte eine grimmige Verw�nschung, setzte
    sich mit einem schauerlichen Grinsen wieder an den Tisch und war bald
    darauf in die Lekt�re des Londoner Polizeiblattes vertieft.

    Oliver befand sich unterdes auf dem Wege zum Buchh�ndler, ohne zu
    ahnen, da� er dem lustigen alten Juden so nahe w�re. Er geriet in eine
    Nebengasse unweit Clerkenwell, bemerkte seinen Irrtum erst, als er sie
    bereits �ber die H�lfte durchwandert hatte, und hielt es f�r das beste,
    um keine Zeit zu verlieren, ihr zu folgen, da sie ihn, wie er meinte,
    auch an sein Ziel f�hren m�sse. Er trabte munter vorw�rts und dachte
    an sein Gl�ck, und was er darum geben w�rde, wenn er den armen kleinen
    Dick daran teilnehmen lassen k�nnte, als er durch den lauten Ruf: �O
    mein lieber kleiner Bruder!� aus seinen Tr�umereien aufgeschreckt
    wurde. Als er aufblickte, umschlossen ihn schon die Arme eines jungen
    M�dchens.

    �Lassen Sie mich los!� rief Oliver, sich str�ubend. �Wer sind Sie? Was
    halten Sie mich an?�

    Die einzige Antwort darauf war ein Schwall lauter Klagen von seiten des
    jungen M�dchens, das einen kleinen Korb und einen Hausschl�ssel in der
    Hand hatte.

    �O g�tiger Himmel!� rief das M�dchen aus. �Endlich hab' ich dich
    gefunden. Ach, Oliver, o du b�ser Junge, was hab' ich um deinetwillen
    ausgestanden! Gott sei Dank, da� ich dich endlich gefunden habe!�

    Das junge Frauenzimmer brach in eine Tr�nenflut aus und schien so
    heftige Kr�mpfe zu bekommen, da� ein paar mitleidige Frauen einen
    dastehenden Fleischerburschen fragten, ob er nicht meinte, da� er zu
    einem Doktor laufen m�sse, worauf der Fleischerbursche, der eine sehr
    gro�e Ruhe, wo nicht ein betr�chtliches Phlegma zu besitzen schien,
    erwiderte, da� seine Meinung nicht dahin ginge.

    �Nein, nein, la�t mich nur�, rief jetzt auch das junge M�dchen; �ich
    f�hle mich schon besser. Und nun komm, mein Junge, geh sogleich mit
    mir, mein b�ser kleiner Liebling.�

    �Was gibt's denn?� fragte eine der umstehenden Frauen.

    �Ach, er ist vor vier Wochen seinen Eltern entlaufen, guten Leuten, die
    sich redlich von ihrer H�nde Arbeit n�hren, und hat sich unter Gauner
    und Landstreicher begeben, da� seine Mutter fast vor Kummer gestorben
    w�re.�

    �O du kleiner Taugenichts! -- Mach, da� du nach Hause kommst, du
    ungeratener Bengel!� riefen die Weiber.

    �Ich bin meinen Eltern nicht entlaufen!� rief Oliver in gro�er Angst.
    �Ich habe weder Schwester noch Eltern. Ich bin eine Waise und wohne in
    Pentonville.�

    �Ach du g�tiger Himmel, wie trotzig er schon geworden ist!� schluchzte
    das junge M�dchen.

    �Ei, Nancy!� rief Oliver, der jetzt erst ihr Gesicht sah, im h�chsten
    Erstaunen aus.

    �Sie sehen, er kennt mich�, sagte Nancy. �Helfen Sie mir ihn nach Hause
    bringen, liebe Leute; seine Eltern und wir alle sterben sonst noch vor
    Kummer �ber ihn.�

    �Zu allen Teufeln, was ist das hier?� schrie ein aus einem Bierladen
    hervorst�rzender Mann. �Oliver, Satansbrut, komm augenblicklich mit
    nach Hause zu deiner armen Mutter. Sofort kommst du mit!�

    �Ich geh�re nicht zu ihnen. Ich kenne sie nicht, Hilfe, Hilfe!� rief
    Oliver, indem er sich unter dem festen Griff des Mannes verzweifelt
    wand.

    �Hilfe!� polterte Sikes. �Ich will dir gleich helfen. Was sind das f�r
    B�cher? -- Ohne Zweifel gestohlen -- her damit!�

    Er entri� ihm das P�ckchen und versetzte ihm damit einen heftigen
    Schlag auf den Kopf.

    �So ist's recht; das wird ihn schon wieder zur Besinnung bringen!�
    riefen die Weiber.

    �Sollt's auch meinen�, rief der Mann, gab Oliver noch ein paar Schl�ge
    auf den Kopf und packte ihn beim Kragen. �Komm, du kleiner Taugenichts!
    Hier, Tyras, pa� auf ihn auf! Pa� auf!�

    Noch geschw�cht von seiner Krankheit, bet�ubt durch die Schl�ge und
    das �berraschende des ganzen Vorganges, in Schrecken gesetzt durch
    das Knurren des Hundes und die Brutalit�t des baumstarken Mannes, und
    �berw�ltigt durch den Beifall, den die Umstehenden seinen Angreifern
    gaben -- was konnte das ge�ngstete Kind tun? Es war dunkel geworden,
    die Gasse sah an sich selbst schon verd�chtig aus, Hilfe war nirgends
    zu erblicken, Widerstand nutzlos. Ohne recht zu wissen, wie ihm
    geschah, f�hlte sich Oliver durch ein Labyrinth von engen Stra�en
    geschleppt, und sein jeweiliges Rufen verhallte um so mehr, da er so
    schnell fortgerissen wurde, da� er keinen Augenblick zu Atem kommen
    konnte; doch w�rde es auch von niemand beachtet worden sein.

           *       *       *       *       *

    Die Gaslampen waren angez�ndet; Frau Bedwin erwartete mit herzpochender
    Ungeduld, da� die Haust�r sich auftun sollte; die Magd war zwanzigmal
    die Stra�e hinuntergelaufen, um nach Oliver auszusehen; die beiden
    alten Herren sa�en beharrlich im Dunkeln neben der zwischen ihnen
    liegenden Uhr.




    16. Kapitel.

        Was sich mit dem entf�hrten Oliver begab.


    Die engen Stra�en und G��chen m�ndeten endlich auf einen weiten,
    offenen Platz, um den rings Stallungen standen zum Zeichen, da�
    hier ein Viehmarkt war. Sikes verlangsamte seinen Schritt, als sie
    diese Gegend erreichten, da das M�dchen v�llig au�erstande war, den
    Laufschritt, den sie bisher angeschlagen hatten, l�nger auszuhalten.
    Sich an Oliver wendend, befahl er ihm barsch, Nancys Hand zu fassen.

    �H�rst du nicht?� brummte Sikes, als Oliver z�gerte und sich umsah.

    Sie befanden sich in einem finsteren, ganz abgelegenen Stadtteil, und
    Oliver sah nur zu gut ein, da� Widerstand nutzlos war. Er streckte
    seine Hand aus, die Nancy fest mit der ihrigen umklammerte.

    Der Abend war dunkel und feucht; die Lichter in den L�den konnten kaum
    gegen den Nebel ank�mpfen, der immer dichter wurde und die Stra�en und
    H�user in ein undurchdringliches Grau h�llte. Sie hatten Smithfield
    erreicht, als tiefe Glockenschl�ge die Stunde verk�ndeten. Sikes und
    Nancy standen bei den ersten Schl�gen still und wandten sich nach der
    Richtung um, aus welcher die T�ne erschallten.

    �Acht Uhr, Bill�, sagte Nancy, als die Glocke aufh�rte zu schlagen.

    �Ich habe selbst Ohren�, erwiderte Sikes m�rrisch.

    �Ich m�chte wohl wissen, ob *sie* es schlagen h�ren k�nnen?� fuhr Nancy
    fort.

    �Nat�rlich k�nnen sie's�, sagte Sikes. �Es war um Bartholom�i, als
    ich in Dobes[L] gesteckt wurde, und auf dem ganzen Markt schnarrte
    keine Pfennigtrompete, die ich nicht geh�rt h�tte. Nachdem ich f�r die
    Nacht eingeschlossen war, machte der L�rm und das Get�se drau�en das
    vermaledeite alte Gef�ngnis so still und einsam, da� ich mir den Kopf
    h�tte einrennen m�gen an den Basteln[M].�

      [L] Gef�ngnis.

      [M] Eisenst�be.

    �Die armen Kerls! Ach, Bill, was sie f�r schmucke junge Leute sind!�

    �Ja, ja, so sprecht ihr Weibsbilder alle!� erwiderte Sikes in einem
    Anfluge von Eifersucht. �Schmucke junge Leute! Doch sie sind so gut wie
    tot, also mag's gleichviel sein.�

    Er fa�te den Knaben wieder fester und trieb zur Eile an.

    �Noch einen Augenblick�, sagte das M�dchen; �ich w�rde nicht
    vorbeilaufen, wenn Ihr's w�r't, der zum Galgen herausgef�hrt w�rde,
    wenn's wieder acht schl�gt. Ich w�rde auf und nieder travallen, bis ich
    nieders�nke, und wenn fu�hoher Schnee l�ge, und ich h�tte kein warmes
    Tuch, mich einzuh�llen.�

    �Das sollte mir wohl viel helfen�, bemerkte der nichtsentimentale
    Sikes. �K�nnt'st du mir nicht � Kulm[N] und � zwanzig Ellen Kabot[O]
    'neinpraktizieren, so m�cht'st du f�nfzig Meilen laufen oder ganz zu
    Hause bleiben, es w�re mir alles nichts n�tze. Vorw�rts, steh' hier
    nicht l�nger und paternelle[P] nicht!�

      [N] Feile.

      [O] Seil, Strick.

      [P] mach keine Predigten.

    Das M�dchen brach in ein Gel�chter aus, ergriff Olivers Hand, und sie
    eilten weiter. Oliver f�hlte, da� ihre Finger zitterten, und als sie an
    einer Gaslampe vor�berkamen, sah er, da� ihr Gesicht totenbla� war.

    Sie lenkten nach einer halben Stunde in eine enge, schmutzige Gasse
    ein, die fast ganz von Tr�dlern bewohnt zu sein schien, und standen
    vor einem verschlossenen Laden still. Das Haus schien unbewohnt zu sein
    und sah halb verfallen aus. �ber der T�r war eine Tafel angenagelt, auf
    welcher zu lesen war, da� das Haus zu vermieten sei; sie schien jedoch
    dort schon jahrelang befestigt gewesen zu sein.

    Nancy b�ckte sich, und Oliver h�rte den Ton einer Glocke. Sie gingen
    auf die entgegengesetzte Seite der Stra�e und stellten sich unter eine
    Laterne. Ein Ger�usch lie� sich h�ren, als ob ein Fenster vorsichtig in
    die H�he geschoben w�rde, und gleich darauf �ffnete sich ger�uschlos
    die T�r. Mr. Sikes packte den erschrockenen Knaben jetzt ohne Umst�nde
    beim Kragen, und im n�chsten Augenblick befanden sich alle drei im
    Innern des Hauses. Hier war es stockfinster. Sie warteten, bis die
    Person, die sie eingelassen hatte, die T�r wieder verschlossen und mit
    einer Sicherheitskette verwahrt hatte.

    �Ist jemand hier?� fragte Sikes.

    �Nein!� erwiderte eine Stimme, die Oliver bekannt vorkam.

    �Ist der Alte hier?� fragte der Dieb.

    �Ja,� antwortete die Stimme, �und er wird sicher sehr erfreut sein, Sie
    zu sehen.�

    �Machen Sie Licht,� versetzte Sikes, �oder wir brechen uns den Hals
    oder treten auf den Hund. Nehmen Sie Ihre Beine in acht, wenn Sie es
    tun.�

    �Bleiben Sie einen Augenblick stehen; ich werde Licht bringen�,
    erwiderte die Stimme. Man h�rte, wie sich der Sprecher entfernte, und
    eine Minute sp�ter erschien die Gestalt John Dawkins', genannt der
    �gepfefferte Baldowerer�. Der junge Herr gab nur durch ein sp�ttisches
    Grinsen kund, da� er Oliver wiedererkannt habe, und bat die Besucher,
    ihm eine Anzahl Stufen hinunter zu folgen. Sie gingen durch eine
    leere K�che und traten in ein niedriges, dumpfiges Gemach ein. Ein
    lautes Gel�chter schallte ihnen entgegen. Charley Bates w�lzte sich
    im eigentlichen Sinne vor Vergn�gen �ber den gar zu kostbaren Spa�
    auf dem Boden, ri� sodann Jack Dawkins das Licht aus der Hand, hielt
    es Oliver dicht vor das Gesicht und beschaute ihn von allen Seiten,
    w�hrend ihm Fagin scherzhafterweise tiefe Verbeugungen machte und der
    Baldowerer, der von ernsterem Wesen war und sich nicht leicht der
    Heiterkeit �berlie�, wenn es Gesch�fte zu verrichten galt, sorgf�ltig
    seine Taschen durchsuchte.

    �Ich freue mich unendlich, Sie so wohl zu sehen, mein Lieber�, sagte
    der Jude. �Der Gepfefferte soll Ihnen geben einen anderen Anzug, damit
    Sie den sonnt�glichen nicht verderben gleich. Warum schrieben Sie's
    nicht, da� Sie kommen wollten -- wir h�tten dann treffen k�nnen noch
    bessere Vorbereitungen -- aber Sie sollen dennoch etwas Warmes bekommen
    zum Abendbrot.�

    Jetzt l�chelte sogar der Baldowerer; da er jedoch in diesem Augenblicke
    die F�nfpfundnote hervorzog, so ist es zweifelhaft, ob der Witz Fagins
    oder die erfreuliche Entdeckung seine Heiterkeit erregte.

    �Holla, was ist das?� rief Sikes und trat auf den Juden zu, als
    derselbe die Banknote hinnahm. �Diese ist mein, Fagin!�

    �Nein, nein, mein Lieber�, entgegnete der Jude. �Mein, Bill, mein; Ihr
    sollt die B�cher haben.�

    �Bekomm' ich und Nancy sie nicht,� sagte Sikes, mit entschlossener
    Miene den Hut aufsetzend, �so bring' ich den Buben wieder zur�ck.�

    Der Jude fuhr empor und Oliver gleichfalls, obgleich aus einem ganz
    anderen Grunde; er hoffte, der Streit w�rde damit enden, da� man ihn
    wieder nach Pentonville zur�ckbr�chte. Allein Sikes entri� dem Juden
    unter Schelten und Drohen die Banknote, faltete sie kaltbl�tig zusammen
    und kn�pfte sie in den Zipfel seines Halstuchs.

    �'s ist f�r unsere M�he und noch nicht halb genug�, sagte er. �Behaltet
    Ihr die B�cher, wenn Ihr gern lest, und wo nicht, schlaget sie los!�

    �Es sind pr�chtige B�cher; nicht wahr, Oliver?� fiel Charley Bates ein,
    als er die kl�gliche Miene gewahrte, mit der Oliver zu seinen Peinigern
    emporblickte.

    �Sie geh�ren dem alten Herrn�, sagte Oliver h�nderingend, �dem lieben,
    guten alten Herrn, der mich in sein Haus nahm und mich pflegen lie�,
    als ich todkrank lag. O bitte, schicken Sie sie zur�ck, schicken Sie
    ihm die B�cher und das Geld zur�ck! Behalten Sie mich hier mein Leben
    lang, aber bitte, bitte, schicken Sie sie nur zur�ck. Er wird glauben,
    da� ich sie gestohlen h�tte -- und die alte Dame und alle, die so
    freundlich gegen mich waren werden es denken. O haben Sie Erbarmen und
    schicken Sie die B�cher und das Geld zur�ck!�

    Oliver fiel vor dem Juden auf die Knie nieder und hob flehend und ganz
    in Verzweiflung die H�nde zu ihm empor.

    �Der Bube hat recht�, sagte Fagin, listig umherblickend und die
    buschigen Augenbrauen zusammenkneifend. �Du hast recht, Oliver, hast
    ganz recht; sie werden allerdings glauben, da� du sie gestohlen hast.
    Ha, ha, ha!� kicherte er und rieb sich die H�nde; �es h�tte sich ganz
    unm�glich treffen k�nnen besser, und wenn wir noch so gut gew�hlt
    h�tten die Zeit.�

    �Versteht sich�, fiel Sikes ein; �ich wu�t's gleich im selbigen
    Augenblick, als ich ihn durch Clerkenwell mit den B�chern unterm
    Arm daherkommen sah. 's ist nun alles gut. Es m�ssen schwachk�pfige
    Betbr�der sein -- h�tten ihn sonst gar nicht zu sich genommen; und sie
    werden auch keine Nachfrage anstellen, aus Furcht, da� sie ihn anklagen
    m��ten und ihn gerumpelt[Q] zu sehen. Wir haben ihn jetzt fest genug.�

      [Q] Auf den Schub bringen -- deportieren.

    Oliver hatte unterdes bald Sikes, bald Fagin angesehen, als wenn er
    ganz bet�ubt w�re und kaum verst�nde, was gesprochen wurde; allein
    bei Bills letzten Worten sprang er pl�tzlich empor und st�rzte unter
    einem Geschrei nach Hilfe aus der T�r hinaus, da� die nackten W�nde des
    Hauses davon widerhallten.

    �Halt den Hund zur�ck, Bill,� schrie Nancy, eilte vor die T�r und
    verschlo� sie, als der Jude mit seinen beiden Z�glingen Oliver
    nachgest�rzt war; �halt den Hund zur�ck; er rei�t ihn in St�cke!�

    �Ist ihm gerade recht!� rief Sikes und suchte sich von dem M�dchen
    loszumachen. �La� mich los, oder ich renn dir den Kopf gegen die Wand!�

    �Ist mir alles gleichviel, Bill, ist mir alles gleichviel�, schrie das
    M�dchen, sich heftig gegen ihn str�ubend; �er soll nicht von dem Hunde
    zerrissen werden, und wenn es mein Tod ist!�

    �So!� tobte Sikes; �sollst nicht lange warten auf deinen Tod, wenn du
    nicht im Augenblick abl�ssest!�

    Er schleuderte sie in die fernste Ecke des Gemachs, gerade als der
    Jude, Jack und Charley den Fl�chtling wieder hereinschleppten.

    �Was gibt's hier?� fragte Fagin.

    �Ich glaube, die Dirne ist toll geworden�, erwiderte Sikes in Wut.

    �Nein, ich bin nicht toll�, rief Nancy bla� und atemlos dazwischen;
    �nein, Fagin, glaubt's nicht!�

    �Dann sei ruhig -- willst du wohl?� sagte der Jude mit drohender
    Geb�rde.

    �Das will ich auch nicht!� erwiderte Nancy mehr schreiend als redend.
    �Was willst du nun?�

    Mr. Fagin war mit den Sitten und Gebr�uchen der Spezies von
    Menschenkindern hinl�nglich bekannt, welcher Mi� Nancy angeh�rte, um
    sich ziemlich �berzeugt zu f�hlen, da� es einigerma�en gef�hrlich sein
    w�rde, die Unterhaltung mit ihr f�r den Augenblick fortzusetzen. Er
    wendete sich daher, um die Aufmerksamkeit der Gesellschaft abzulenken,
    zu Oliver.

    �Du wolltest also fortlaufen, mein Lieber?� sagte er, einen Knotenstock
    aufhebend, der am Kamine lag; �wolltest rufen die Polizei -- nicht
    wahr, mein Schatz? Ich will dich von der Krankheit kurieren, lieber
    Engel!�

    Er hatte bei diesen Worten Oliver beim Arme gefa�t, versetzte ihm einen
    Schlag �ber den R�cken und hob den Knotenstock wieder empor, als Nancy
    auf ihn zust�rzte, ihm den Stock aus der Hand ri� und in das Feuer
    schleuderte.

    �Ich leid's nimmermehr, Fagin!� schrie sie. �Ihr habt den Knaben, und
    was wollt Ihr mehr? La�t ihn -- la�t ihn zufrieden, oder ich tue etwas
    an Euch, das mich vor meiner Zeit an den Galgen bringt!� Sie stampfte
    bei dieser Drohung heftig mit den F��en und blickte mit verbissenen
    Lippen, geballten F�usten und bla� vor Zorn und Wut abwechselnd den
    Juden und Sikes an.

    �Ah, Nancy!� sagte der Jude nach einer kurzen, verlegenen Pause
    beschwichtigend; �du -- du �bertriffst dich wirklich heute abend selbst
    -- ha, ha, ha! -- spielst ganz prachtvoll deine Rolle, liebes Kind!�

    �So!� entgegnete Nancy; �nehmt Euch nur in acht, da� ich sie nicht zu
    gut f�r Euch spiele. Ich sage es Euch vorher, Ihr werdet Euch sehr
    schlecht dabei stehen!�

    Es gibt wenige M�nner, die sich nicht gern enthielten, ein in Wut
    geratenes und obendrein von nichtsachtender Verzweiflung beseeltes
    Frauenzimmer noch mehr zu reizen. Der Jude sah ein, da� es ihm nichts
    helfen k�nne, sich noch l�nger zu stellen, als wenn er Nancys Zorn
    f�r blo� erk�nstelt hielte, fuhr unwillk�rlich einige Schritte zur�ck
    und blickte halb zitternd, halb verzagend nach Sikes. Dieser mochte
    glauben, sein pers�nliches Ansehen fordere es, Nancy baldigst wieder
    zur Vernunft zu bringen, und begann daher seine Operationen mit
    zahlreichen und kr�ftigen Drohungen und Verw�nschungen, wobei er den
    Beweis lieferte, da� er es in diesem Genre in der Tat zur Meisterschaft
    gebracht hatte. Als sie keinen sichtbaren Eindruck machten, ging er zu
    noch �berzeugenderen Argumenten �ber. �Was soll das bedeuten, Dirne?�
    tobte er unter Hinzuf�gung einer Verw�nschung, die die Blindheit so
    gew�hnlich als die Masern machen w�rde, wenn der Himmel sie nur halb
    so oft wahr machte, als man sie auf Erden h�rt. �Was willst du damit
    bezwecken? Wei�t du, zum Geier, wer du bist -- was du bist?�

    �O ja, ja; ich wei� es nur zu gut!� erwiderte Nancy unter krampfhaftem
    Lachen, dabei den Kopf hin und her wiegend, um gleichg�ltig zu
    erscheinen, was ihr jedoch schlecht gelang.

    �Dann sei ruhig, oder ich werde dich auf 'ne lange Zeit zum
    Stillschweigen bringen.�

    Sie lachte abermals, blickte fl�chtig nach Sikes, wendete das Gesicht
    ab und bi� sich die Lippen blutig.

    �Du bist mir die Rechte, dich auf die menschenfreundliche und honette
    Seite zu legen!� fuhr er ver�chtlich fort. �Der Bursch w�rde 'ne
    saubere Freundin an dir haben, wozu du dich aufwirfst!�

    �Und beim allm�chtigen Gott, ich bin es!� rief sie mit
    leidenschaftlicher Heftigkeit; �und ich wollte lieber, da� ich auf der
    Stra�e tot niedergefallen oder in das Gef�ngnis geworfen w�re, statt
    derer, denen wir so nahe waren, als da� ich mich dazu hergegeben h�tte,
    ihn hierher zu bringen. Er ist von heute abend an ein Dieb, ein L�gner,
    ein M�rder, ein Teufel und alles, was nur schlecht und verworfen hei�en
    mag; -- ist das nicht genug f�r den alten Halunken -- mu� er ihn
    obendrein schlagen?�

    �H�rt, Bill,� fiel der Jude dringend und nach dem mit gespanntem
    Ohr zuh�renden Knaben hindeutend ein, �wir m�ssen freundliche Worte
    gebrauchen, freundliche Worte, Bill.�

    �Freundliche Worte!� schrie das in seiner Wut schrecklich aussehende
    M�dchen; �freundliche Worte, Ihr Schuft! Ja, die verdient Ihr auch von
    mir! Ich habe gestohlen f�r Euch, als ich noch nicht halb so alt war
    wie dies Kind hier, und bin in demselben Gesch�ft und demselben Dienst
    seit zw�lf Jahren gewesen; wi�t Ihr das nicht? Sprecht, wi�t Ihr es
    nicht?�

    �Ja, ja doch�, erwiderte der Jude bes�nftigend; �du hast ja aber auch
    davon dein Brot.�

    �Freilich, ich habe mein Bettelbrot davon,� schrie sie immer heftiger,
    �und die kalten, nassen, schmutzigen Stra�en sind meine Wohnung; und
    Ihr seid der ruchlose Mann, der mich Tag und Nacht hinaustreibt und
    mich Tag und Nacht hinaustreiben wird, bis ich im Grabe liege.�

    �Ich f�ge dir ein Leid zu,� versetzte der Jude, durch diese Vorw�rfe
    gereizt, �ein Leid, das schlimmer ist als das, von dem du sprichst,
    wenn du noch ein Wort sagst.�

    Nancy erwiderte nichts mehr, zerraufte aber in einem �berma� von
    Leidenschaft ihr Haar, st�rzte auf Fagin zu, und auf seinem Gesichte
    w�rden ohne Zweifel sichtbare Spuren ihrer Rache zur�ckgeblieben sein,
    h�tte nicht Sikes eben noch zur rechten Zeit ihre Arme festgehalten.
    Sie bem�hte sich vergeblich, sich von ihm loszurei�en, und sank in
    Ohnmacht. �Es ist nun alles wieder in Ordnung�, bemerkte Sikes, sie in
    eine Ecke tragend. �Sie besitzt au�erordentliche K�rperkr�fte, wenn sie
    sich in diesem Zustand befindet.� Der Jude wischte sich die Stirn und
    l�chelte, und sowohl er wie Sikes und die Knaben schienen den ganzen
    Vorfall als einen gew�hnlichen, im Gesch�ft h�ufig vorkommenden zu
    betrachten.

    �Es ist doch das Schlimmste, mit Weibern zu tun zu haben�, bemerkte
    der Jude, den Stock wieder beiseite stellend; �aber sie sind schlauer
    als wir, und wir k�nnen ohne sie nicht fertig werden. Charley, bringe
    Oliver zu Bett.�

    �Nicht wahr, Fagin, er soll morgen seine besten Kleider nicht tragen?�
    fragte Charley Bates grinsend, und der Jude verneinte, Charleys
    liebliches Grinsen erwidernd. Master Bates schien sich seines Auftrages
    h�chlich zu freuen, f�hrte Oliver in das ansto�ende Gemach, in welchem
    einige Betten der Art standen, wie er sie bereits kennen gelernt, und
    zog mit unbezwinglichem Gel�chter die alten Kleidungsst�cke hervor, die
    sich Oliver so gefreut hatte, ablegen zu d�rfen, und die Fagin auf die
    erste Spur seines Aufenthaltes bei Mr. Brownlow gebracht hatten.

    �Zieh' die Sonnt�gischen aus,� sagte Charley, �ich will sie Fagin zum
    Aufheben geben. Welch' ein pr�chtiger Spa�!�

    Der arme Oliver gehorchte widerstrebend und wurde darauf von Charley im
    Dunkeln gelassen und eingeschlossen. Master Bates' Gel�chter und die
    Stimme Betsys, die nach einiger Zeit erschien, und ihre Freundin zum
    Bewu�tsein zur�ckzurufen sich bem�hte, w�ren gar wohl geeignet gewesen,
    ihn unter anderen Umst�nden wach zu erhalten; allein er war ersch�pft
    und unwohl und schlief daher bald ein.




    17. Kapitel.

        Olivers Schicksal bleibt fortw�hrend g�nstig.


    In jedem guten Melodrama, in dem viel von Hauen und Stechen die Rede
    ist, wechseln auf der B�hne komische und tragische Szenen so regelm��ig
    wie die roten und wei�en Lagen eines St�cks durchwachsenen Specks.
    Diese Abwechselungen erscheinen uns abgeschmackt, sind indes keineswegs
    unnat�rlich. Die �berg�nge im wirklichen Leben von wohlbesetzten
    Tischen zu Sterbebetten oder von Trauer- zu Festtagskleidern sind nicht
    minder schroff oder gef�hlverletzend -- wir aber sind besch�ftigte
    Mitspielende statt blo�er Zuschauer, was einen unerme�lichen
    Unterschied bildet; den Schauspielern sind die pl�tzlichen �berg�nge
    nicht auff�llig, sie haben sozusagen keine Augen f�r dieselben, die
    von den Zuschauern verkehrt, unnat�rlich, extravagant genannt werden.
    Verdamme mich daher nicht zu voreilig, geneigter Leser, wenn du in
    meinem Buche einen h�ufigen Wechsel des Schauplatzes und der Szenen
    findest, sondern erzeige mir die Gunst, zu pr�fen, ob ich recht oder
    unrecht dabei gehabt habe. Meine Erz�hlung soll meiner Absicht nach
    wahr sein und ohne unn�tige Abschweifungen auf ihr Ziel lossteuern. Ich
    bitte, folge mir f�r jetzt vertrauensvoll nach der Stadt, in welcher
    mein kleiner Held das Licht der Welt erblickte.

    Mr. Bumble trat eines Morgens fr�h aus dem Armenhause mit der
    wichtigsten Miene heraus, und durchschritt die Stra�en mit einer
    Haltung und einem Wesen, da� man es ihm sogleich ansah, sein Inneres
    war von Gedanken erf�llt, zu gro�, um sie aussprechen zu k�nnen. Er
    hielt sich nicht unterwegs auf, um sich mit den kleinen Kr�mern und
    anderen, die ihn anredeten, in herablassender Weise zu unterhalten,
    sondern erwiderte ihre Begr��ungen nur mit einer hoheitsvollen
    Handbewegung, und hemmte seinen w�rdevollen Schritt erst, als er vor
    der Anstalt stand, in der Mrs. Mann die Armenkinder mit parochialer
    Sorgfalt pflegte.

    �Dieser verw�nschte Kirchspieldiener!� sagte Mrs. Mann zu sich selbst,
    als sie das bekannte R�tteln an der Pforte h�rte. �Ob er nicht schon in
    aller Herrgottsfr�he herauskommt! Schau, Mr. Bumble, soeben habe ich an
    Sie gedacht. Ja, verehrter Herr, es ist mir ein wirkliches Vergn�gen,
    Sie wieder einmal zu sehen! Treten Sie, bitte, n�her.�

    Der erste Satz war zu Susanne gesprochen worden, die Freudenbezeigungen
    dagegen zu Mr. Bumble, als die gute Dame die Gartenpforte �ffnete und
    ihn mit gro�er H�flichkeit und Ehrerbietung ins Haus n�tigte.

    �Mrs. Mann,� sagte Mr. Bumble, indem er sich mit gro�er Feierlichkeit
    und W�rde auf einen Stuhl niederlie�, �Mrs. Mann, Ma'am, ich biete
    Ihnen einen guten Morgen.�

    �Ich danke Ihnen und biete Ihnen auch meinerseits einen guten Morgen�,
    erwiderte Mrs. Mann freundlich l�chelnd; �ich hoffe, Sie befinden sich
    wohl, Sir.�

    �So -- so, Mrs. Mann,� antwortete der Kirchspieldiener; �man ist in der
    Parochie nicht immer auf Rosen gebettet.�

    �Ach ja, das ist man in der Tat nicht�, versetzte die Dame, und alle
    Armenkinder w�rden ihr laut beigepflichtet haben, falls sie ihre Worte
    geh�rt h�tten.

    �Ein Leben im Dienste der Parochie�, fuhr Mr. Bumble fort, �ist ein
    Leben voller M�hseligkeiten und Plagen, Ma'am; aber alle �ffentlichen
    Charaktere, darf ich wohl sagen, m�ssen unter Verfolgungen leiden.�

    Mrs. Mann, die nicht genau wu�te, was der Kirchspieldiener meinte,
    erhob ihre H�nde mit einem Seufzer des Einverst�ndnisses.

    �Ach ja,� bemerkte Mr. Bumble, �Sie haben wohl ein Recht zu seufzen,
    Ma'am.�

    Da Mrs. Mann fand, sie habe richtig gehandelt, seufzte sie von neuem,
    offenbar zur Befriedigung des ��ffentlichen Charakters�; denn Mr.
    Bumble sagte, ein wohlgef�lliges L�cheln unterdr�ckend: �Mrs. Mann, ich
    gehe nach London.�

    �Was Sie sagen, Mr. Bumble!� erwiderte Mrs. Mann erstaunt.

    �Nach London, Ma'am,� wiederholte der Kirchspieldiener
    unersch�tterlich, �und zwar in einer Postkutsche. Ich und zwei Arme,
    Mrs. Mann.�

    �Sie benutzen eine Postkutsche, Sir?� fragte Mrs. Mann. �Ich glaubte,
    es w�re immer �blich, Arme auf offenen Karren zu verschicken.�

    �Das geschieht, wenn sie krank sind�, entgegnete der Kirchspieldiener;
    �wir setzen die kranken Armen bei Regenwetter auf offene Karren, damit
    sie sich nicht erk�lten. Die Postkutsche nimmt diese beiden au�erdem
    um ein sehr Billiges mit, und wir finden, es kommt uns um zwei Pfund
    wohlfeiler zu stehen, wenn wir sie in ein anderes Kirchspiel schaffen
    k�nnen, als wenn wir sie hier begraben m�ssen. Hahaha! -- Aber wir
    vergessen das Gesch�ft, Ma'am,� fuhr er ernst werdend fort; �hier ist
    das Kostgeld f�r den Monat.�

    Mr. Bumble holte eine kleine Rolle mit Silbergeld aus seiner
    Brieftasche hervor und bat um eine Quittung; Mrs. Mann schrieb sie
    sofort.

    �Ich danke Ihnen recht sehr, Mr. Bumble; ich bin Ihnen in der Tat f�r
    Ihre Liebensw�rdigkeit sehr verbunden.�

    Mr. Bumble nickte gn�dig in Anerkennung der H�flichkeit Mrs. Manns, und
    erkundigte sich sodann nach dem Befinden der Kinder.

    �Gott segne ihre lieben kleinen Herzchen�, erwiderte Mrs. Mann bewegt;
    �sie befinden sich den Umst�nden angemessen wohl, die lieben Kleinen!
    Nat�rlich bis auf die zwei, die vergangene Woche gestorben sind, und
    den kleinen Dick.�

    �Geht es mit dem Jungen immer noch nicht besser?� fragte Mr. Bumble.
    Mrs. Mann sch�ttelte den Kopf.

    �Er ist ein schlecht beanlagtes, lasterhaftes Parochialkind mit �blen
    Angewohnheiten�, sagte Mr. Bumble �rgerlich. �Wo ist er?�

    �Ich werde ihn Ihnen in einer Minute herbringen, Sir�, gab Mrs. Mann
    zur Antwort.

    Nach einigem Suchen wurde Dick entdeckt und nach gr�ndlicher S�uberung
    unter der Pumpe Mr. Bumble vorgef�hrt.

    Das Kind war bleich und mager; seine Wangen waren eingesunken und seine
    Augen gro� und fieberisch gl�nzend. Die armselige Parochialkleidung,
    die Livree seines Elends, hing schlotternd um seinen schw�chlichen
    K�rper, und seine jungen Glieder waren welk wie die eines alten Mannes.

    �Wie geht es dir?� fragte Mr. Bumble den Knaben, der zitternd dastand
    und seine Augen nicht vom Fu�boden zu erheben vermochte.

    �Ich glaube, da� ich bald sterben mu�,� erwiderte der kleine Patient,
    �und ich freue mich auch recht darauf, denn ich habe ja keine Freude
    hier. Sagen Sie doch Oliver Twist, wenn ich erst tot bin, ich h�tte
    ihn sehr lieb gehabt und tausendmal an ihn gedacht, wie er allein und
    hilflos umherwandern m��te --�

    Er hatte die Worte mit einer Art von Verzweiflung gesprochen, ohne
    sich durch Frau Manns pantomimische Drohungen irren zu lassen; doch
    erstickten endlich Tr�nen seine Stimme.

    �Frau Mann,� bemerkte Bumble, �ich sehe wohl, der eine ist wie der
    andere. Sie sind samt und sonders durch den Taugenichts Oliver Twist
    verf�hrt und verdorben worden. Ich werde dem Direktorium Anzeige von
    dem Falle machen, damit strengere Ma�regeln angeordnet werden. Lassen
    Sie ihn sogleich wieder hinausbringen!�

    Dick wurde in den Kohlenkeller gebracht, und Bumble begab sich wieder
    zur Stadt zur�ck, wo er sich in k�rzester Frist reisefertig machte
    und mit den beiden nach London zu schaffenden Armen die bestellten
    Au�enpl�tze der Postkutsche einnahm. Die beiden Armen klagten viel
    �ber K�lte; Bumble h�llte sich dicht in seinen Mantel, philosophierte
    ziemlich mi�vergn�gt �ber den Undank und die unabl�ssigen unzufriedenen
    Klagen der Menschen und f�hlte sich erst wieder recht behaglich, als er
    in dem Gasthause, in welchem die Kutsche anhielt, sein gutes Abendessen
    eingenommen, seinen Stuhl an den Kamin gestellt hatte, sich niederlie�
    und ein Zeitungsblatt zur Hand nahm. Wer beschreibt sein Erstaunen, als
    er gleich darauf nachstehenden Artikel fand:

    �*F�nf Guineen Belohnung*.

    Am vergangenen Donnerstag abend hat sich ein Knabe namens Oliver
    Twist aus seiner Wohnung in Pentonville entfernt und mit oder ohne
    seine Schuld nichts wieder von sich h�ren lassen. Es werden hierdurch
    demjenigen f�nf Guineen geboten, der eine Mitteilung zu machen geneigt
    und imstande ist, die zur Wiederauffindung des besagten Oliver Twist
    f�hren kann, oder �ber denselben, seine Herkunft usw. genauere Auskunft
    gibt.�

    Diesem Anerbieten folgte eine genaue Beschreibung Olivers und Mr.
    Brownlows Adresse. Bumble las dreimal mit gro�em Bedacht, fa�te darauf
    rasch seinen Entschlu� und war nach wenigen Minuten auf dem Wege nach
    Pentonville. Im Hause Mr. Brownlows angelangt, k�ndigte er sogleich
    den Zweck seines Besuchs an. Frau Bedwin war au�er sich vor Freude und
    R�hrung, erkl�rte, es immer gewu�t und gesagt zu haben, da� Oliver bald
    wiedergefunden werden w�rde, brach in Tr�nen aus, und die Magd eilte
    zu Mr. Brownlow hinauf, der ihr gebot, den Angemeldeten augenblicklich
    hereinzuf�hren.

    Bumble trat ein, und Mr. Grimwig, der sich zuf�llig bei seinem
    Freunde befand, fa�te ihn scharf in das Auge und rief aus: �Ein
    Kirchspieldiener -- so wahr ich lebe, ein Kirchspieldiener!�

    �Ich bitte, lieber Freund, jetzt keine Unterbrechung�, sagte Brownlow.
    �Setzen Sie sich, Sir. -- Sie kommen zu mir infolge der Anzeige, die
    ich in verschiedenen Bl�ttern habe einr�cken lassen?�

    �Ja, Sir.�

    �Sie sind ein Kirchspieldiener?�

    �Ja, Sir�, erwiderte Bumble stolz.

    �Wissen Sie, wo sich das arme Kind gegenw�rtig befindet?� fragte
    Brownlow ziemlich ungeduldig.

    �Nein, Sir.�

    �Was wissen Sie denn aber von ihm? Reden Sie, wenn Sie etwas zu sagen
    haben. Was wissen Sie von ihm?�

    �Sie werden wohl eben nicht viel Gutes von ihm wissen!� fiel Grimwig
    kaustisch ein, nachdem er Bumbles Mienen sorgf�ltig gepr�ft hatte.

    Bumble erkannte sogleich mit gro�em Scharfsinn den Wunsch des Herrn,
    Ung�nstiges �ber Oliver zu vernehmen, und antwortete durch ein
    feierlich-bedenkliches Kopfsch�tteln.

    �Sehen Sie wohl?� sagte Grimwig zu Brownlow mit einem triumphierenden
    Blicke.

    Brownlow sah Bumble besorgt an, und forderte ihn auf, was er von
    Oliver w��te, in m�glichst kurzen Worten mitzuteilen. Bumble r�usperte
    sich und begann. Er sprach mit umst�ndlicher Weitschweifigkeit; der
    kurze Sinn von allem, was er vorbrachte, war, Oliver sei ein armer
    Kirchspielknabe von armen und lasterhaften Eltern, habe von seiner
    Geburt an nur Falschheit, Bosheit und Undankbarkeit gezeigt und seiner
    Gottlosigkeit dadurch die Krone aufgesetzt, da� er einen m�rderischen
    und feigherzigen Angriff auf einen harmlosen Knaben gemacht habe und
    darauf seinem Lehrherrn entlaufen sei.

    �Ich f�rchte, da� Ihre Angaben nur zu wahr sind�, sagte Brownlow
    traurig; �hier sind die f�nf Guineen. Ich w�rde Ihnen gern dreimal so
    viel gegeben haben, wenn Sie mir etwas Vorteilhafteres �ber den Knaben
    h�tten sagen k�nnen.�

    H�tte Brownlow das fr�her gesagt, so w�rde Bumble seinem Bericht
    wahrscheinlich eine ganz andere F�rbung gegeben haben. Es war jedoch zu
    sp�t, er sch�ttelte daher mit bedeutsamer Miene den Kopf, steckte die
    f�nf Guineen ein und ging.

    Mr. Brownlow war so niedergeschlagen, da� selbst Grimwig ihn nicht
    noch mehr betr�ben mochte. Er zog endlich heftig die Klingelschnur.
    �Frau Bedwin,� sagte er, als die Haush�lterin eintrat, �der Knabe, der
    Oliver, war ein Betr�ger.�

    �Das kann nicht sein, Sir; kann nicht sein�, entgegnete Frau Bedwin
    nachdr�cklich.

    �Ich sage Ihnen aber, da� es so ist. Wir haben soeben einen genauen
    Bericht �ber ihn angeh�rt. Er ist von seiner ersten Kindheit an durch
    und durch verderbt gewesen.�

    �Und ich glaube es doch nicht, Sir -- nimmermehr, Sir�, erwiderte Frau
    Bedwin bestimmt.

    �Ihr alten Weiber glaubt an nichts als an Quacksalber und
    L�gengeschichten�, fiel Grimwig m�rrisch ein. �Ich hab's von Anfang an
    gewu�t. Warum h�rten Sie nicht sogleich auf meine Meinung und meinen
    Rat? Sie w�rden es sicher getan haben, wenn der kleine Schelm nicht am
    Fieber krank gelegen h�tte!�

    �Er war kein Schelm,� entgegnete Frau Bedwin sehr unwillig, �sondern
    ein sehr liebes, gutes Kind. Ich verstehe mich auf Kinder sehr wohl,
    Sir, seit vierzehn Jahren, Sir; und wer nie Kinder gehabt hat, darf gar
    nicht mitreden �ber sie -- das ist meine Meinung, Sir!�

    Mr. Grimwig l�chelte nur, und Frau Bedwin war im Begriff, fortzufahren,
    allein Brownlow kam ihr zuvor.

    �Schweigen Sie!� sagte er mit einer Entr�stung in Ton und Mienen, die
    freilich seinen Gef�hlen vollkommen fremd war. �Sie erw�hnen den Knaben
    nie wieder; ich habe geklingelt, um Ihnen das zu sagen. H�ren Sie --
    nie -- niemals, und unter keinerlei Vorwande. Sie k�nnen gehen -- und
    wohl zu merken, ich habe im Ernst gesprochen!�

    In Mr. Brownlows Hause waren betr�bte Herzen an diesem Abende, und
    Oliver zagte das Herz gleichfalls, als er seiner g�tigen Besch�tzer und
    Freunde gedachte. Es war indes gut f�r ihn, da� er nicht wu�te, was sie
    �ber ihn geh�rt; er h�tte die Nacht vielleicht nicht �berlebt.




    18. Kapitel.

        Wie Oliver seine Zeit in der sittenverbessernden Gesellschaft
        seiner achtungsw�rdigen Freunde zubrachte.


    Als am folgenden Morgen der Baldowerer und Charley Bates zu ihren
    gew�hnlichen Gesch�ften ausgegangen waren, benutzte Mr. Fagin die
    Gelegenheit, Oliver einen langen Sermon �ber die schreiende S�nde
    der Undankbarkeit zu halten, deren er sich, wie ihm Fagin kl�rlich
    bewies, in einem sehr hohen Ma�e schuldig gemacht, indem er sich
    absichtlich von seinen liebevollen und treuen Freunden entfernt, ja
    sogar ihnen zu entfliehen versucht habe, nachdem sie so viele M�he
    und Kosten aufgewandt h�tten, ihn wieder zur�ckzubringen. Der alte
    Herr legte gro�es Gewicht auf den Umstand, da� er Oliver zu sich
    genommen und verpflegt habe, als derselbe in Gefahr gewesen w�re,
    Hungers zu sterben, und erz�hlte ihm die ergreifende und schreckliche
    Geschichte eines jungen Burschen, dem er unter �hnlichen Umst�nden
    aus gewohnter Menschenfreundlichkeit seinen Beistand habe angedeihen
    lassen, der sich aber des ihm erwiesenen Vertrauens unw�rdig gezeigt,
    sich mit der Polizei in Rapport zu setzen versucht habe und im
    Old-Bailey-Gerichtshofe verurteilt und gehenkt worden sei. Der alte
    Herr bem�hte sich durchaus nicht, seinen Anteil an der Katastrophe
    zu verheimlichen, sondern beklagte es mit Tr�nen in den Augen,
    da� es durch die Verkehrtheit und Verr�terei des jungen Burschen
    n�tig geworden, ihn als ein Opfer fallen zu lassen, und demnach mit
    Zeugnissen gegen ihn aufzutreten, die, wenn auch nicht vollkommen
    in der Wahrheit begr�ndet, doch unumg�nglich gewesen w�ren, wenn
    seine (Fagins) und einiger erlesener Freunde Sicherheit nicht h�tte
    gef�hrdet werden sollen. Der alte Herr schlo� damit, da� er ein sehr
    unerfreuliches Gem�lde von den Unannehmlichkeiten des Gehenktwerdens
    entwarf und mit gro�er Freundschaftlichkeit und H�flichkeit die
    Hoffnung ausdr�ckte, niemals gen�tigt zu werden, Oliver Twist einer so
    widerw�rtigen Operation zu unterwerfen.

    Dem kleinen Oliver erstarrte das Blut in den Adern, w�hrend er den
    Worten des Juden zuh�rte. Die darin enthaltenen dunklen Drohungen waren
    ihm nicht ganz unverst�ndlich. Er wu�te bereits, da� die Gerechtigkeit
    selbst den Unschuldigen f�r schuldig halten konnte, wenn er sich mit
    dem Schuldigen in Gemeinschaft befunden hatte; und da� tief angelegte
    Pl�ne, unbequeme Mitwisser oder zum Schwatzen Geneigte zu verderben,
    von dem alten Juden wirklich geschmiedet und ausgef�hrt w�ren, d�nkte
    ihn keineswegs unwahrscheinlich, als er sich des Streites entsann, den
    Fagin mit Sikes gehabt. Als er furchtsam die Augen aufschlug und seine
    Blicke denen des Juden begegneten, f�hlte er, da� seine Bl�sse und
    sein Zittern dem schlauen B�sewicht nicht entgangen waren und da� sich
    derselbe innerlich dar�ber freute.

    Der Jude l�chelte boshaft, klopfte Oliver die Wangen und sagte ihm,
    wenn er sich ruhig verhielte und sich des Gesch�fts ann�hme, so w�rden
    sie sicher noch sehr gute Freunde werden. Er griff darauf zum Hute, zog
    einen alten, geflickten Oberrock an, ging hinaus und verschlo� die T�r
    hinter sich.

    So blieb sich Oliver w�hrend des ganzen Tages und w�hrend noch vieler
    nachfolgender Tage vom fr�hen Morgen bis Mitternacht selbst �berlassen,
    und die langen Stunden vergingen ihm gar traurig, denn er gedachte
    nat�rlich fortw�hrend seiner g�tigen Freunde in Pentonville und der
    Meinung, welche sie von ihm gefa�t haben m��ten. Am siebenten oder
    achten Tage lie� der Jude die T�r des Zimmers unverschlossen, und
    Oliver durfte frei im Hause umhergehen. -- Das ganze Haus war �u�erst
    schmutzig und �de; die Zimmer im oberen Stockwerke waren ohne Mobilien,
    geschw�rzt und mit Spinngeweben �berdeckt; indes schlo� Oliver aus dem
    T�felwerke und den Resten alter Tapeten und anderer Verzierungen, da�
    sie vor langer Zeit von reichen Leuten bewohnt gewesen sein m��ten,
    so kl�glich sie auch jetzt aussahen. Oft, wenn er leise in ein Zimmer
    eintrat, liefen die M�use erschreckt in ihre L�cher zur�ck; sonst aber
    sah und h�rte er kein lebendiges Wesen, und manches Mal, wenn er es
    m�de war, aus einem Gemach in das andere zu wandern, schmiegte er sich
    in den Winkel des Flurs an der Haust�r, um den Menschen so nahe wie
    m�glich zu sein, und erwartete horchend und mit Beben die R�ckkehr des
    Juden oder der Knaben.

    In allen Zimmern waren die Fensterl�den fest mit Schrauben verwahrt und
    lie�en nur wenig Licht durch kleine, runde L�cher ein, was die Zimmer
    noch d�sterer machte und sie mit seltsamen Schattengestalten f�llte.
    Ein hinteres Dachst�bchen hatte ein mit starken St�ben verwahrtes
    Fenster ohne L�den. Oliver schaute stundenlang traurig hinaus, obwohl
    er nichts sehen konnte als eine verworrene, gedr�ngte Masse von
    D�chern, geschw�rzten Schornsteinen und Giebeln. Bisweilen zeigte sich
    auf ein paar Augenblicke in der Dachluke eines fernen Hauses ein nur
    undeutlich zu erkennendes Gesicht; allein es verschwand bald, und da
    das Fenster von Olivers Observatorium vernagelt und durch Regen und
    Rauch von Jahren tr�b und blind gemacht worden war, so war es ihm nur
    m�glich, die Formen ferner Gegenst�nde undeutlich zu erkennen, und er
    konnte nicht daran denken, sich bemerkbar zu machen, zumal auch die
    Nachbarschaft sicher nicht die achtbarste und vertrauenerweckendste war.

    Eines Nachmittags kehrten der Baldowerer und Charley Bates nach Hause
    zur�ck, um sich auf eine Abendunternehmung vorzubereiten, die es
    erfordern mochte, da� sie sich sorgf�ltiger als gew�hnlich ankleideten.
    Der Baldowerer gebot Oliver, ihm die Stiefel zu reinigen, und Oliver
    war froh, nur einmal Menschen zu sehen und sich n�tzlich machen zu
    k�nnen, wenn es ohne Verletzung der Redlichkeit geschehen konnte.
    W�hrend er besch�ftigt war, dem Gehei� Folge zu leisten, wobei Jack
    auf einem Tische sa�, blickte der junge Gentleman zu ihm hernieder,
    seufzte und sagte halb zerstreut und halb zu Charley Bates: �'s ist
    doch Jammer und Schade, da� er kein Kochemer ist.�

    �Ah,� sagte Charley Bates, �er wei� nicht, was ihm gut ist.�

    �Du wei�t wohl nicht mal, Oliver, was ein Kochemer ist?� fragte der
    Baldowerer.

    �Ich glaube es zu wissen�, erwiderte Oliver sch�chtern; �ein Dieb --
    bist du nicht ein Dieb?�

    �Ja,� sagte Jack, �und ich rechn' es mir zur Ehre. Ich bin � Dieb;
    Charley ist's, Fagin ist's, Sikes ist's; Nancy und Betsy sind
    gleichfalls Diebinnen. Wir sind samt und sonders Diebe, bis herunter zu
    Sikes Hund, und der geht noch �ber uns alle.�

    �Und ist kein Angeber�, bemerkte Charley Bates.

    �Er w�rde in der Zeugenloge nicht mal bellen, um sich nicht zu verraten
    oder verd�chtig zu machen�, fuhr Jack fort. �Doch das hat nichts zu
    schaffen mit unserm Musj� Gr�n.�

    �Warum begibst du dich nicht unter Fagins Oberbefehl, Oliver?� fiel
    Charley ein.

    �K�nntest doch dein Gl�ck so sch�n machen�, setzte Jack hinzu, �und
    von deinem Gelde leben wie ein Gentleman, wie ich's zu tun denke im
    n�chstkommenden f�nften Schaltjahr und am zweiundvierzigsten Dienstag
    in der Fastenwoche.�

    �Es gef�llt mir nicht�, sagte Oliver furchtsam. �Ich wollte, da� Fagin
    mich fortgehen lie�e.�

    �Das wird Fagin bleiben lassen�, bemerkte Charley.

    Oliver wu�te dies nur zu gut, und in der Meinung, es m�chte gef�hrlich
    sein, seine Gedanken noch offener auszusprechen, fuhr er seufzend in
    seinem Gesch�fte fort.

    �Sch�me dich!� hub der Baldowerer wieder an. �Hast du denn gar kein
    Ehrgef�hl? Ich m�chte um nichts in der Welt meinen Freunden zur Last
    fallen, am wenigsten, ohne 'nen Finger zu r�hren, um ihnen zum
    wenigsten meine Erkenntlichkeit zu beweisen.�

    �Es ist wahrhaftig zu gemein und niedrig�, sagte Charley Bates, einige
    seidene Taschent�cher hervorziehend und in eine Kommode legend.

    �Es w�re mir ganz unm�glich!� rief Jack Dawkins, sich in die Brust
    werfend, aus.

    �Und doch k�nnt ihr eure Freunde im Stich lassen�, bemerkte Oliver mit
    einem halben L�cheln, �und zusehen, da� sie f�r Dinge bestraft werden,
    die ihr getan habt.�

    �Es geschah blo� aus R�cksicht gegen Fagin�, erwiderte Jack kaltbl�tig.
    �Die Schoderer[R] wissen, da� wir gemeinschaftlich arbeiten, und er
    h�tte in Ungelegenheit kommen k�nnen, wenn wir nicht davongelaufen
    w�ren. Schau hier�, setzte er hinzu, griff in die Tasche und zeigte
    Oliver eine Handvoll Schillinge und Halbpence. �Wir f�hren � flottes
    Leben, und was tut's, woher das Geld dazu kommt? Da, nimm hin; wo's
    her ist, da ist noch mehr von der Sorte. Du willst nicht? O D�mmling,
    D�mmling aller D�mmlinge!�

      [R] Gerichtsdiener.

    �Er ist � B�sewicht, nicht wahr, Oliver?� fiel Charley Bates ein. �Er
    wird noch geschn�rt werden, nicht wahr?�

    �Ich wei� nicht, was das ist�, sagte Oliver.

    Charley Bates nahm sein Taschentuch, kn�pfte es sich um den Hals und
    stellte die H�ngeoperation pantomimisch und vollkommen kunstgerecht
    dar. �Das ist's�, sagte er endlich unter schallendem Gel�chter.

    �Du bist schlecht erzogen,� bemerkte Jack Dawkins ernsthaft; �indes
    wird Fagin doch schon noch etwas aus dir machen, oder du w�rst der
    erste, der sich ganz unbrauchbar gezeigt. Also fang nur, je eher, desto
    lieber, an, denn du wirst mitarbeiten im Gesch�ft, eh' du's meinst, und
    verlierst nur Zeit, Oliver.�

    Charley Bates f�gte noch mehrere moralische Betrachtungen hinzu,
    schilderte mit gl�henden Farben die zahllosen Annehmlichkeiten des
    Lebens, das sie, er und Jack, f�hrten, und bem�hte sich mit einem Worte
    auf das eifrigste, Oliver zu �berzeugen, da� er nichts Besseres tun
    k�nne, als baldm�glichst um Fagins Gunst durch dieselben Mittel zu
    werben, die er und Jack zum gleichen Zwecke angewendet.

    �Und vor allen Dingen, Nolly[S],� sagte Jack, als sie den Juden kommen
    h�rten, �bedenk' das: nimmst du keine Schneichen und Zwiebeln --�

      [S] Oliver.

    �Was hilft's, da� du so zu ihm redest?� unterbrach Charley; �wei� er
    doch nicht, was du damit sagen willst!�

    �Nimmst du keine Taschent�cher und Uhren,� fuhr der Baldowerer, zu
    Olivers Fassungskraft sich herablassend, fort, �so tut's der erste
    beste andere, und der hat was davon, und du hast nischt, da du doch ein
    ebenso gutes Recht dazu hast.�

    �'s ist ganz klar -- ja, ja -- ganz klar,� sagte der Jude, der
    unbemerkt von Oliver eingetreten war, �klar wie die Sonne, mein Kind.
    Glaub' dem Baldowerer; er kennt den Katechismus seines Gesch�fts aufs
    Haar.�

    Das Gespr�ch wurde indes f�r jetzt abgebrochen, da Fagin mit Mi� Betsy
    und einem Gentleman angelangt war, den Oliver noch nicht gesehen hatte
    und den der Baldowerer Tom Chitling nannte, als er eintrat, nachdem
    er drau�en ein wenig verweilt, um mit der Dame einige Galanterien zu
    wechseln.

    Tom Chitling war �lter an Jahren als der Baldowerer, da er etwa
    achtzehn Winter z�hlen mochte, bezeigte demselben aber eine
    Ehrerbietung, woraus man kl�rlich sah, da� er sich bewu�t war, an
    Genie und Gesch�ftserfahrung ihm untergeordnet zu sein. Tom hatte
    kleine, blinzelnde Augen und ein pockennarbiges Gesicht und trug eine
    Pelzkappe, eine Jacke aus dunklem Tuch, fettige Barchenthosen und eine
    Sch�rze. Er sah in der Tat ziemlich abgerissen aus, entschuldigte sich
    jedoch bei der Gesellschaft damit, da� �seine Zeit� erst seit einer
    Stunde aus gewesen sei, da� er seine Uniform sechs Wochen getragen
    und noch nicht daran habe denken k�nnen, die Garderobe zu wechseln.
    Er schlo� mit der Bemerkung, da� er zweiundvierzig Tage angestrengt
    gearbeitet, und �bersten wolle, wenn er in der ganzen Zeit 'nen Tropfen
    gekostet und nicht so trocken sei wie ein Sandfa߻.

    �Was meinst du, Oliver, woher der junge Mensch wohl kommt?� fragte der
    Jude grinsend, w�hrend Charley eine Branntweinflasche auf den Tisch
    stellte.

    �Ich -- ich kann's nicht sagen, Sir�, erwiderte Oliver.

    �Wer ist denn der?� fragte Tom Chitling, Oliver ver�chtlich anblickend.

    �Ein junger Freund von mir, mein Lieber�, antwortete Fagin.

    �Dann hat er's gut genug�, bemerkte Tom, dem Juden einen bedeutsamen
    Blick zuwerfend. �K�mmere dich nicht darum, Bursch, woher ich komme; es
    gilt 'ne Krone, wirst bald genug selber da sein!�

    Es wurde gelacht, Fagin fl�sterte mit Tom, alle versammelten sich
    am Kamine, der Jude forderte Oliver auf, sich zu ihm zu setzen, und
    lenkte das Gespr�ch auf Gegenst�nde, von welchen er erwarten konnte,
    da� seine Zuh�rer den lebhaftesten Anteil daran nehmen w�rden; n�mlich
    die gro�en Vorteile des Gesch�fts, die Talente des Baldowerers, die
    Liebensw�rdigkeit Charleys und die Freigebigkeit Fagins. Als sie
    ersch�pft waren und Tom Chitling gleichfalls Zeichen der Ersch�pftheit
    an den Tag legte (denn das Besserungshaus erm�det sehr nach einigen
    Wochen), entfernte sich Mi� Betsy, und die �brigen begaben sich zur
    Ruhe.

    Von diesem Tage an wurde Oliver nur noch selten allein gelassen und
    in eine fortw�hrende enge Verbindung mit Jack und Charley gebracht,
    die mit dem Juden t�glich das alte Spiel spielten -- Fagin wu�te am
    besten, ob zu ihrer eigenen oder Olivers Belehrung und Vervollkommnung.
    Zu anderen Zeiten erz�hlte ihnen Fagin Geschichten von Diebst�hlen und
    R�ubereien, die er in seinen j�ngeren Tagen begangen, und mischte so
    viel Merkw�rdiges, Spa�haftes und Drolliges ein, da� Oliver sich oft
    nicht enthalten konnte, herzlich zu lachen und den Beweis zu liefern,
    da� er trotz seiner besseren Gef�hle Wohlgefallen an diesen Geschichten
    fand.

    Kurzum, der schlaue alte Jude hatte den Knaben sozusagen im Netze und
    war, nachdem er ihn durch Einsamkeit und die Qual derselben dahin
    gebracht, jede Gesellschaft seinen traurigen Gedanken in einem so �den,
    finsteren Hause vorzuziehen, eifrig dar�ber aus, seinem Herzen das Gift
    langsam einzufl��en, das, wie er hoffte, die Farbe desselben ver�ndern
    und es f�r immer schw�rzen sollte.




    19. Kapitel.

        In welchem ein verh�ngnisvoller Plan besprochen und beschlossen
        wird.


    Es war ein kalter, feuchter und st�rmischer Abend, als der Jude seinen
    eingeschrumpften Leib in einen Oberrock einh�llte, den Kragen �ber die
    Ohren zog, so da� von seinem Gesicht nur die Augen zu sehen waren, und
    sich aus seiner H�hle entfernte. Er blieb vor der Haust�r stehen, bis
    sie inwendig verschlossen und verriegelt war, und eilte darauf mit
    leisen und fl�chtigen Schritten die Stra�e hinunter.

    Das Haus, in welches Oliver gebracht worden war, befand sich nahe bei
    Whitechapel; der Jude stand an der n�chsten Ecke ein paar Augenblicke
    still, schaute forschend umher und schlug sodann die Richtung nach
    Spitalfields ein.

    Auf dem Pflaster lag dicker Schlamm, und ein dichter Nebel machte die
    Dunkelheit noch dunkler. F�r den Ausflug eines d�monischen Wesens,
    wie es der Jude war, konnten Zeit, Wetter und alle Umgebungen nicht
    passender sein. Der greuliche Alte glich, w�hrend er verstohlen durch
    Nacht und Nebel und Kot dahineilte, einem ekelhaften Gew�rm, das in
    n�chtlicher Finsternis aus seinem Verstecke herauskriecht, um w�hlend
    im Schlamme ein leckeres Mahl nach seiner Art zu halten.

    Er setzte seinen Weg durch viele enge und winklige Gassen fort, bis er
    Bethnal Green erreichte, wandte sich dann nach links und verschwand in
    einem wahrhaften Labyrinth schmutziger Winkel, Stra�en und Gassen jenes
    zahlreich bev�lkerten Stadtviertels, ohne jedoch ein einziges Mal zu
    irren oder fehlzugehen, lenkte endlich in eine Sackgasse ein, klopfte
    an die T�r eines Hauses und wurde, nachdem er ein paar Worte durch das
    Schl�sselloch gefl�stert, eingelassen und hinaufgef�hrt.

    Als er auf den Griff einer T�r fa�te, knurrte ein Hund, und eine grobe
    Mannsstimme fragte, wer da w�re.

    �Ich bin's, Bill, ich, mein Lieber�, antwortete der Jude hineinschauend.

    �So bringt Eur'n Leichnam 'rein�, sagte Sikes. �Lieg' still, dumme
    Bestie! Kennst den Teufel nicht, wenn er'n �berrock anhat?�

    Der Hund schien in der Tat durch Fagins Verh�llung get�uscht zu sein;
    denn sobald der Jude den Oberrock aufkn�pfte, legte er sich, mit dem
    Schweife wedelnd, wieder nieder.

    �Nun?� sagte Sikes.

    �Ja -- nun�, erwiderte der Jude. �Ah, Nancy.�

    Er schien etwas verlegen und zweifelhaft zu sein, wie er von Mi� Nancy
    empfangen werden w�rde, denn er hatte seine junge Freundin seit dem
    Abend noch nicht wiedergesehen, an welchem sie so leidenschaftlich f�r
    Oliver aufgetreten war. Das Benehmen der jungen Dame machte jedoch
    bald aller Ungewi�heit ein Ende. Sie schob ihren Stuhl zur Seite und
    forderte Fagin auf, ohne Groll oder noch viel Worte sich mit an den
    Kamin zu setzen, denn es w�re ein kalter Abend.

    �Ja, 's ist bitter kalt, liebe Nancy�, sagte Fagin und begann seine
    kn�chernen H�nde �ber dem Feuer zu w�rmen. �'s ist, als wenn der Wind
    einem wehte durch und durch bis ins Innerste.�

    �Das mu� wirklich scharf sein, was bis an dein Herz dringt�, bemerkte
    Sikes. �Gib ihm 'nen Tropfen zu trinken, Nancy. Alle Donnerwetter, mach
    geschwind! Man wird ganz �bel davon, das alte Gerippe so schaudern zu
    sehn wie'n h��liches Gespenst, das eben aus'm Grabe gestiegen ist.�

    Nancy holte schnell eine Flasche aus dem Schranke; Sikes schenkte ein
    Glas Branntwein ein und hie� den Juden es austrinken; Fagin ber�hrte es
    jedoch nur mit den Lippen und setzte es wieder auf den Tisch.

    �Ausgetrunken, Spitzbube!� rief Sikes.

    �Habe schon genug, danke, Bill!�

    �Wie -- was? F�rchtest dich, da� wir dir � Streich spielen?� fragte
    Sikes, seine Augen scharf auf den Juden richtend.

    Mit einem heiseren, ver�chtlichen Brummen ergriff Mr. Sikes das Glas
    und go� den Inhalt in die Asche; dann f�llte er es von neuem und
    st�rzte es hinunter.

    Fagin blickte im Zimmer umher, nicht aus Neugierde, denn es war ihm
    wohlbekannt, sondern unruhig, verstohlen, argw�hnisch, wie es ihm zur
    Gewohnheit geworden war. Das Gemach war sehr schlecht m�bliert. Nur der
    Inhalt des Schrankes schien anzudeuten, da� es von einem gew�hnlichen
    Arbeiter bewohnt w�rde; auch sah man nichts Verd�chtiges, mit Ausnahme
    einiger schwerer Kn�ttel, die in einem Winkel standen, und eines
    �Lebensretters�, der �ber dem Kaminsimse hing.

    �Was hast du zu sagen, verdammter Jude?� fragte Sikes. �Weshalb bist du
    hergeschlichen?�

    �Wegen des Bayes[T] in Chertsey, Bill�, erwiderte der Jude, dicht zu
    ihm r�ckend und fl�sternd.

      [T] Haus.

    �Nun -- und was weiter?�

    �Ah -- Ihr wi�t ja recht gut, was ich meine, Bill. Nicht wahr, Nancy,
    er wei� es recht gut?�

    �Nein, er wei� es nicht�, fiel Sikes h�hnisch ein, �oder will es nicht
    wissen, was dasselbe ist. Sprich rein 'raus, nenn' die Dinge beim
    rechten Namen und stell' dich nicht an, als wenn du nicht der erste
    gewesen w�rst, der an den Einbruch gedacht hat.�

    �Pst, Bill, pst!� sagte Fagin, der sich vergebens bem�ht hatte, Sikes
    zum Stillschweigen zu bringen; �es wird uns jemand h�ren, mein Lieber,
    es wird uns jemand h�ren!�

    �La� h�ren, wer will!� tobte Sikes; �'s ist mir alles gleich.�

    Er sprach jedoch die letzten Worte schon weniger laut und heftig, da
    ihm der Gedanke kam, da� es doch *nicht* gleich w�re oder sein k�nnte.

    �Seid doch ruhig, Bill,� sagte der Jude bes�nftigend. �Es war ja nur
    meine Vorsicht -- weiter nichts. Also wegen des Bayes in Chertsey, mein
    Lieber. Wann soll's sein, Bill -- wann soll's sein? Solch Silberzeug,
    Bill, solch Silberzeug!� setzte er h�ndereibend und mit leuchtenden
    Augen hinzu.

    �Gar nicht�, erwiderte Sikes trocken.

    �Gar nicht?� wiederholte der Jude und lehnte sich erstaunt auf seinem
    Stuhle zur�ck.

    �Nein, gar nicht�, sagte Sikes; �zum wenigsten kann's nicht so
    ausgef�hrt werden, wie wir meinten.�

    �Dann ist's nicht geschickt und ordentlich angegriffen�, versetzte der
    Jude, vor Verdru� erblassend. �Aber Ihr spa�t nur, Bill.�

    �Ich lasse mich lieber h�ngen, als da� ich mit dir spa�e, altes
    Gerippe. Toby Crackit hat sich seit vierzehn Tagen die erdenklichste
    M�he gegeben, aber keinen von der Dienerschaft --�

    �Ihr wollt doch nicht sagen, Bill,� unterbrach ihn der Jude ungeduldig,
    doch aber ruhiger in dem Ma�e, als Sikes wieder heftig zu werden
    anfing; �Ihr wollt doch nicht sagen, da� keiner von den beiden
    Bedienten k�nnte werden gewonnen, zu machen Kippe?�

    �Das will ich allerdings sagen�, antwortete Sikes. �Sie sind seit
    zwanzig Jahren bei der alten Frau im Dienst gewesen und w�rden's nicht
    tun f�r f�nfhundert Pfund.�

    �Aber die weibliche Dienerschaft, mein Lieber -- l��t sich die auch
    nicht beschwatzen?�

    �Nein!�

    �Wie -- auch nicht vom schmucken, geriebenen Toby Crackit?� entgegnete
    der Jude ungl�ubig. �Bedenkt doch nur, wie die Weibsen sind, Bill!�

    �Nein, auch nicht von Toby Crackit�, erwiderte Sikes. �Er hat die ganze
    Zeit, da� er's Bayes umschlichen, falsche Knebelb�rte und 'ne gelbe
    Weste getragen; hat aber alles nicht helfen wollen.�

    �Er h�tt's versuchen sollen mit 'nem Schnurrbart und Soldatenhosen,
    mein Lieber�, sagte der Jude nach einigem Besinnen.

    �Das hat er auch schon getan, und 's ist ebenso vergeblich gewesen.�

    Der Jude machte eine verdrie�liche und verlegene Miene dazu, versank
    auf ein paar Minuten in tiefes Nachsinnen und sagte endlich mit einem
    schweren Seufzer, wenn man sich auf Toby Crackits Berichte verlassen
    k�nnte, so f�rchte er, da� der Plan aufgegeben werden m�sse. �'s ist
    aber sehr betr�bend, Bill,� setzte er, die H�nde auf die Knie st�tzend,
    hinzu, �so viel zu verlieren, wenn man einmal den Sinn hat gesetzt
    darauf.�

    �Freilich,� sagte Sikes, �'s ist ganz verdammt �rgerlich!�

    Es folgte ein langes Stillschweigen. Der Jude war in tiefe Gedanken
    verloren, und sein Gesicht nahm einen Ausdruck wahrhaft satanischer
    Spitzb�berei an. Sikes blickte ihn von Zeit zu Zeit verstohlen von
    der Seite an, und Nancy heftete, aus Furcht, den Wohnungsinhaber
    zu erz�rnen, die Augen auf das Feuer, als wenn sie bei allem, was
    gesprochen worden, taub gewesen w�re.

    �Fagin,� unterbrach Sikes endlich die allgemeine Stille, �schafft's
    f�nfzig F�chse extra, wenn's durch Einbruch vollbracht wird?�

    �Ja!� rief der Jude, wie aus einem Traume erwachend.

    �Abgemacht?� fragte Sikes.

    �Ja, mein Lieber,� erwiderte der Jude, indem er ihm die Hand reichte;
    und jede Muskel seines Gesichts gab Zeugnis, wie freudig und lebhaft
    er durch diese Frage �berrascht worden war.

    Sikes schob die Hand des Juden ver�chtlich zur�ck und fuhr fort: �Dann
    mag's geschehen, sobald du willst, Alter. Toby und ich sind vorgestern
    �ber die Gartenmauer g'wesen und haben die T�ren und Fensterl�den
    untersucht. Die Bayes ist nachts verrammelt wie'n Dobes; wir haben
    aber 'ne Stelle gefunden, wo wir leise und mit Sicherheit schr�nken[U]
    k�nnen.�

      [U] einbrechen.

    �Wo ist denn die Stelle, Sikes?� fragte der Jude sehr gespannt.

    �Man geht �ber den Rasenplatz,� fl�sterte Sikes, �und dann --�

    �Nun, und dann?� unterbrach ihn der Jude, sich ungeduldig vorbeugend.

    �Dann --� sagte der Schr�nker, brach jedoch kurz ab, denn Nancy gab
    ihm, kaum den Kopf bewegend, einen Wink, nach des Juden Gesicht zu
    sehen. �'s ist ganz gleich, wo die Stelle ist�, fuhr er fort. �Ich
    wei�, da� du's nicht kannst ohne mich; aber man tut wohl daran, sich
    auf Nummer Sicher zu setzen, wenn man mit dir zu tun hat.�

    �Nach Eurem Belieben, Bill, nach Eurem Belieben�, erwiderte der Jude,
    sich auf die Lippen bei�end. �K�nnt Ihr's mit Toby allein, und braucht
    Ihr weiter keinen Beistand?�

    �Nein; blo� ein Dreheisen und 'nen Knaben. Das Eisen haben wir, den
    Buben mu�t du uns schaffen.�

    �'nen Knaben!� rief der Jude aus. �Ah, dann ist's ein Paneel -- wie?�

    �Es kann dir gleichviel sein, was es ist�, erwiderte Sikes. �Ich
    brauche 'nen Buben, und er darf nicht gro� sein. Wenn mir nur nicht der
    von Ned, dem Schornsteinfeger, durch die Lappen 'gangen w�re! Er hielt
    ihn mit Absicht klein und schm�chtig und lieh ihn aus f�r'n Billiges.
    Aber so geht's, der Vater wird gerumpelt[V], und wie der Blitz ist
    der Verein f�r verlassene Kinder da und nimmt den Jungen aus 'nem
    Gesch�ft, darin er Geld h�tte verdienen k�nnen, lehrt ihn Lesen und
    Schreiben, und der Bube wird dann Lehrling, Gesell, endlich Meister,�
    sagte Sikes mit steigendem Zorn �ber einen so unrechtm��igen Verlauf,
    �und so geht's mit den meisten; und h�tten sie immer Geld genug, was
    sie Gott Lob und Dank nicht haben, so w�rden wir nach ein paar Jahren
    keinen einzigen Jungen mehr im Gesch�ft halten.�

      [V] deportiert.

    �Ja, ja�, stimmte der Jude ein, der unterdes �berlegt und nur die
    letzten Worte geh�rt hatte. �Bill!�

    �Was gibt's?�

    Der Jude deutete verstohlen auf Nancy hin, die noch immer in das Feuer
    schaute, und gab Sikes durch Zeichen seinen Wunsch zu erkennen, mit ihm
    allein gelassen zu werden. Sikes zuckte ungeduldig die Achseln, als wenn
    er die Vorsicht f�r �berfl�ssig hielte, forderte indes Nancy auf, ihm
    einen Krug Bier zu holen.

    �Ihr seid nicht durstig, Bill�, sagte Nancy mit der vollkommensten Ruhe
    und schlug die Arme �bereinander.

    �Ich sage dir, ich bin durstig!� entgegnete Sikes.

    �Dummes Zeug! Fahrt fort, Fagin. Ich wei�, was er sagen will, Bill; ich
    kann's auch h�ren.�

    Der Jude z�gerte, und Sikes sah etwas verwundert bald ihn, bald das
    M�dchen an.

    �Brauchst dich vor dem alten M�dchen nicht zu scheuen, Fagin�, sagte er
    endlich. �Hast sie lange genug gekannt und kannst ihr trauen, oder der
    Teufel m��te drin sitzen. Sie wird nicht mosern[W]; nicht wahr, Nancy?�

      [W] verraten.

    �Ihr sollt's wohl meinen�, erwiderte sie, ihren Stuhl an den Tisch
    schiebend und den Kopf auf die Ellbogen st�tzend.

    �Nein, nein, liebes Kind�, fiel der Jude ein; �ich wei� das sehr wohl;
    nur --� Er hielt wieder inne.

    �Nun, was denn nur?� fragte Sikes.

    �Ich wei� nur nicht, ob sie nicht vielleicht wieder werden w�rde
    unwirsch, mein Lieber, wie vor einigen Abenden�, erwiderte Fagin.

    Bei diesem Gest�ndnisse brach Nancy in ein lautes Gel�chter aus,
    st�rzte ein Glas Branntwein hinunter, erkl�rte unter mehrfachen
    kr�ftigen Beteuerungen, da� sie alles h�ren k�nne, wolle und werde und
    so standhaft, mutvoll und treu sei wie eine oder einer. -- �Fagin,�
    sagte sie lachend, �sprecht nur ohne Umschweife zu Bill von Oliver!�

    �Ah! Du bist ein so gewitztes M�dchen, wie ich je eins gesehen�,
    versicherte der Jude und klopfte sie auf die Wange. �Ja, ich wollte
    wirklich sprechen von Oliver; ha, ha, ha!�

    �Was ist mit ihm los?� fragte Sikes.

    �Da� er der Knabe ist, den Ihr braucht, mein Lieber�, erwiderte der
    Jude in einem heiseren Fl�stern, den Finger an die Nase legend und mit
    einem f�rchterlichen Grinsen.

    �Der Oliver?!� rief Bill aus.

    �Nimm ihn, Bill�, sagte Nancy. �Ich t�t's, wenn ich an deiner Stelle
    w�re. Mag sein, da� er nicht so gepfifft und dreist ist wie einer
    von den andern; aber das ist auch nicht n�tig, wenn du ihn blo� dazu
    brauchen willst, da� er dir 'ne T�r aufmacht. Verla� dich darauf, er
    ist petacht[X], Bill.�

      [X] zuverl�ssig.

    �Ich wei�, da� er's ist�, fiel Fagin ein. �Er ist in den letzten
    Wochen geschult gut, und 's ist Zeit, da� er anf�ngt, f�r sein Brot zu
    arbeiten; au�erdem sind die andern alle zu gro�.�

    �Ja, die rechte Gr��e hat er�, bemerkte Sikes nachdenklich.

    �Und er wird alles tun, wozu Ihr ihn n�tig habt, Bill�, sagte der Jude.
    �Er kann nicht anders -- n�mlich, wenn Ihr ihn nur genug haltet in
    Furcht und Schrecken.�

    �Das k�nnte geschehen -- und nicht blo� zum Spa�. Ist was nicht richtig
    mit ihm, wenn wir einmal erst am Werk sind -- alle Teufel! -- so siehst
    du ihn nicht lebendig wieder, Fagin. Bedenk' das, eh' du ihn schickst.�

    Er hatte ein schweres Brecheisen unter dem Bette hervorgezogen und
    sch�ttelte es unter drohenden Geb�rden.

    �Ich habe alles bedacht�, erwiderte der Jude entschlossen. �Ich hab'
    ihn beobachtet, meine Lieben, wie ein Falke die Augen auf ihn gehabt.
    La�t ihn nur erst wissen, da� er einer der Unsrigen ist; la�t ihn nur
    erst wissen, da� er gewesen ist ein Dieb, und er ist unser -- unser auf
    sein Leben lang! Oho! Es h�tte nicht besser k�nnen kommen!� Er kreuzte
    die Arme �ber der Brust, zog den Kopf zwischen die Schultern und
    umarmte sich gleichsam selbst vor Behagen und Freude.

    �Unser!� h�hnte Sikes. �Du willst sagen: dein.�

    �K�nnte vielleicht sein, mein Lieber�, sagte der Jude kichernd. �Wenn
    Ihr's so wollt, Bill, mein.�

    Sikes warf seinem angenehmen Freunde finster-grollende Blicke zu. �Und
    warum bem�hst du dich denn so sehr um das Kreidegesicht,� sagte er,
    �da du doch wei�t, da� jede Nacht f�nfzig Buben im Common Garden[Y]
    dormen[Z], unter denen du die Wahl hast?�

      [Y] Coventgarden.

      [Z] schlafen.

    �Weil ich sie nicht gebrauchen kann, mein Lieber�, erwiderte der
    Jude ein wenig verwirrt. �Sie sind's nicht wert, da� man's versucht
    mit ihnen, denn wenn sie in Ungelegenheiten geraten, steht ihnen
    geschrieben auf der Stirn, was sie sind und was sie haben getan,
    und sie gehen mir alle kapores. Aber mit diesem Knaben, wenn er nur
    gebraucht wird geschickt, kann ich ausrichten mehr als mit zwanzig
    von den anderen. Au�erdem,� f�gte er wieder in vollkommener Fassung
    hinzu, �au�erdem *haben wir* ihn dann fest jetzt, wenn er uns wieder
    entwischen k�nnte, und *er* mu� bleiben mit uns im selben Boot,
    gleichviel wie er gekommen ist hinein; ich habe Macht genug �ber ihn,
    wenn er nur ein einziges Mal ist gewesen bei 'nem Schr�nken -- mehr
    brauch' ich nicht. Und wieviel ist das besser, als wenn wir m��ten den
    armen, kleinen Knaben �ber die Seite schaffen, was w�rde gef�hrlich
    sein -- und wodurch wir verlieren w�rden viel!�

    Sikes schwebte eine starke Mi�billigung bei Fagins pl�tzlicher
    Anwandlung von Menschlichkeit auf den Lippen, Nancy kam ihm jedoch
    durch die Frage zuvor, wann der Einbruch geschehen sollte.

    �Ja, Bill, ja -- wann soll es sein?� fragte auch der Jude.

    �Ich hab's mit Toby auf �bermorgen nacht verabredet,� antwortete Sikes
    m�rrisch, �wenn ich ihm keine anderweitige Nachricht zugehen lasse.�

    �Gut�, sagte der Jude; �es wird doch kein Mondschein sein?�

    �Nein�, erwiderte Sikes.

    �Ist auch bedacht alles wegen Fortschaffens der Sechore[AA]?� fragte
    Fagin.

      [AA] Gestohlenes Gut.

    Sikes nickte.

    �Und wegen --�

    �Ja, ja, 's ist alles verwaldiwert[AB],� unterbrach ihn Sikes; �scher
    dich nur nicht weiter drum. Bring' den Buben lieber morgen abend her.
    Ich werde 'ne Stunde nach Tagesanbruch auf und davon sein. Und dann
    halt's Maul und stelle den Schmelztiegel bereit; das ist alles, was du
    zu tun hast.�

      [AB] verabredet.

    Nach einigem Hin- und Herreden, woran alle drei t�tigen Anteil nahmen,
    wurde beschlossen, da� Nancy am folgenden Abend Oliver herbringen
    solle. Fagin hielt daf�r, da� er Nancy am ersten folgen w�rde, wenn er
    etwa abgeneigt w�re. Ebenso wurde feierlich verabredet, da� der Knabe
    zum Zweck der beabsichtigten Unternehmung Sikes unbedingt �bergeben
    werden solle, und zwar so, da� derselbe mit ihm nach Gutd�nken
    verfahren d�rfe, ohne dem Juden f�r irgendeinen Unfall, der ihn treffen
    k�nnte, oder irgendeine Z�chtigung verantwortlich zu sein, die sein
    Besch�tzer etwa f�r notwendig erachten m�chte; auch sollte der letztere
    alle seine Angaben nach seiner R�ckkehr durch Toby Crackits Zeugnis
    best�tigen lassen. Sikes bekr�ftigte vorl�ufig den edlen Bund und
    die Aufrichtigkeit seiner Gesinnungen durch ein Glas Branntwein nach
    dem andern, was die Wirkung hatte, da� er zuerst l�rmte und sodann
    einschlief. Der Jude h�llte sich darauf wieder in seinen �berrock,
    sagte Nancy gute Nacht und fa�te sie scharf ins Auge, w�hrend sie ihm
    zur Erwiderung gleichfalls wohl zu schlafen w�nschte und ihre Blicke
    den seinigen begegneten. Sie waren vollkommen fest und ruhig. Das
    M�dchen war so treu und verl��lich in der Sache, wie Toby Crackit nur
    selbst sein konnte. Er warf Sikes, unbemerkt von ihr, noch einen Blick
    des Hasses und der Verachtung zu und ging, durch die Z�hne murmelnd:
    �So sind sie alle. Das Schlimmste an den Weibsbildern ist, da� die
    gr��te Kleinigkeit aufweckt in ihnen ein l�ngst vergessenes Gef�hl --
    und das Beste, da� es nicht w�hrt lange. Hi, hi, hi! Ich wette 'nen
    Sack voll Gold auf den Mann gegen das Kind.�

    Unter diesen angenehmen Gedanken ging Fagin seines Weges durch Schmutz
    und Kot hin bis zu seiner d�steren Wohnst�tte. Der Baldowerer war
    aufgeblieben und erwartete ungeduldig die R�ckkehr des Juden.

    �Ist Oliver zu Bett? Ich w�nsche ihn zu sprechen�, war die erste Frage,
    die er tat, als beide die Treppe hinunterstiegen.

    �Schon seit mehreren Stunden�, versetzte der Baldowerer, indem er eine
    T�r aufstie�. �Hier ist er.�

    Der Knabe lag fest schlafend auf einer harten Matratze auf dem
    Fu�boden, so bleich vor Angst, Traurigkeit und Verlassenheit in seinem
    Gef�ngnis, da� er �hnlichkeit mit einem Toten hatte -- nicht mit einem
    Toten, wie er im Sarge und auf der Bahre liegt, sondern mit einem,
    aus dem das Leben soeben entwichen ist, wenn ein junger, edler Geist
    zum Himmel entflohen ist und die schwere Luft der Welt noch keine
    Zeit gefunden hat, den zarten Schimmer, von dem er umgeben war, zu
    verdr�ngen.




    20. Kapitel.

        In welchem Oliver Sikes �berliefert wird.


    Als Oliver am folgenden Morgen erwachte, war er nicht wenig verwundert,
    ein Paar neue Schuhe mit starken, dicken Sohlen an der Stelle seiner
    alten, sehr besch�digten zu erblicken. Anfangs freute er sich der
    Entdeckung, weil er sie als eine Vorl�uferin seiner Befreiung ansah;
    allein er gab bald alle Gedanken dieser Art auf, als er sich allein mit
    dem Juden zum Fr�hst�ck setzte, der ihm, und zwar auf eine Weise, die
    ihn mit Unruhe erf�llte, sagte, da� er am Abend zu Bill Sikes gebracht
    werden solle.

    �Soll -- soll ich denn dort bleiben?� fragte Oliver angstvoll.

    �Nein, nein, Kind, du sollst nicht dort bleiben�, antwortete der Jude.
    �Wir w�rden dich gar nicht gern missen. Sei ohne Furcht, Oliver; du
    sollst wieder zur�ckkehren zu uns. Ha, ha, ha! Wir werden nicht sein
    so grausam, dich fortzuschicken, mein Kind. Nein, beileibe nicht!� Der
    alte Mann, der sich �ber das Feuer geb�ckt hatte und eine Brotschnitte
    r�stete, sah sich bei diesen sp�ttischen Worten um und kicherte, wie
    um zu zeigen, er wisse es, da� Oliver gern entfliehen w�rde, wenn er
    k�nnte.

    �Ich glaube, Oliver,� fuhr er, die Blicke auf ihn heftend, fort, �du
    m�chtest wissen, weshalb du sollst zu Bill -- nicht wahr, mein Kind?�

    Oliver verf�rbte sich unwillk�rlich, denn er gewahrte, da� der Jude in
    seinem Innern gelesen, erwiderte indes dreist, da� er es allerdings zu
    wissen w�nsche.

    �Nun, was meinst du wohl, weshalb?� fragte Fagin, der Antwort
    ausweichend.

    �Ich kann es nicht erraten, Sir�, erwiderte Oliver.

    �Pah! So warte, bis Bill dir's sagt�, versetzte Fagin, sich mi�vergn�gt
    abwendend, denn er hatte in Olivers Mienen wider Verhoffen nichts
    entdeckt, nicht einmal Neugierde.

    Die Wahrheit ist indessen, da� der Knabe allerdings sehr lebhaft zu
    wissen w�nschte, zu welchem Zwecke er Sikes �berliefert werden sollte,
    aber durch Fagins forschende Blicke und sein eigenes Nachsinnen zu sehr
    au�er Fassung geraten war, um f�r den Augenblick noch weitere Fragen
    zu tun. Sp�ter fand sich keine Gelegenheit dazu, denn der Jude blieb
    bis gegen Abend, da er sich zum Ausgehen anschickte, sehr m�rrisch und
    schweigsam.

    �Du kannst brennen ein Licht�, sagte er und stellte eine Kerze auf den
    Tisch; �und da ist ein Buch, worin du kannst lesen, bis sie kommen,
    dich abzuholen. Gute Nacht!�

    �Gute Nacht, Sir�, erwiderte Oliver sch�chtern.

    Der Jude ging nach der T�r und sah �ber die Schulter nach dem Knaben
    zur�ck; dann stand er pl�tzlich still und rief ihn beim Namen.

    Oliver blickte auf, der Jude wies nach dem Lichte hin und befahl ihm,
    es anzuz�nden. Oliver tat, wie ihm gehei�en wurde, und sah, da� Fagin
    mit gerunzelter Stirn aus dem dunkleren Teile des Gemachs forschend die
    Augen auf ihn heftete.

    �H�te dich, Oliver, h�te dich!� sagte der Alte, warnend die rechte
    Hand emporhebend. �Er ist ein brutaler Mann und achtet kein Blut, wenn
    seins ist hei�. Was sich auch zutr�gt, sprich kein Wort und tu', was
    er dir sagt. Nimm dich in acht! -- wohl in acht!� Er hatte die letzten
    Worte mit scharfer Betonung gesprochen, sein finsterer, drohender Blick
    verwandelte sich in ein greuliches L�cheln, er nickte und ging.

    Oliver legte den Kopf auf die Hand, als er allein war, und sann mit
    pochendem Herzen den eben vernommenen Worten nach. Je l�nger er �ber
    die Warnung des Juden nachdachte, in eine desto gr��ere Ungewi�heit
    geriet er �ber ihren eigentlichen Sinn und Zweck. Er konnte sich nichts
    B�ses oder Unrechtes bei seiner Sendung zu Sikes denken, das nicht
    ebensogut geschehen oder erreicht werden konnte, wenn er bei Fagin
    blieb. Nach langem Nachsinnen kam er zu dem Schlusse, da� er ausersehen
    sein m�chte, Sikes als Aufw�rter zu dienen, bis man einen besser dazu
    geeigneten Knaben gefunden h�tte. Er war zu sehr an Leiden und Dulden
    gew�hnt und hatte zu viel gelitten in dem Hause, in welchem er sich
    befand, als da� ihn die Aussicht auf eine Ver�nderung des Schauplatzes
    seiner Widerw�rtigkeiten sehr h�tte betr�ben k�nnen. Er blieb noch eine
    Weile in Gedanken verloren, putzte seufzend das Licht und fing an in
    dem Buche zu lesen, das ihm der Jude zur�ckgelassen.

    Er hatte anfangs nur gebl�ttert, allein eine Stelle erregte seine
    Aufmerksamkeit im h�chsten Grade, und bald las er um so eifriger. Das
    Buch enthielt Erz�hlungen von ber�chtigten Verbrechern aller Art und
    trug auf jeder Seite die Spuren eines sehr h�ufigen Gebrauchs. Er las
    hier von furchtbaren Verbrechen, die das Blut zu Eis erstarren lie�en,
    von Raubmorden, die auf offener Landstra�e ver�bt worden waren, von
    Leichen, die man vor den Augen der Menschen in den tiefen Brunnen
    und Sch�chten verborgen hatte, ohne da� es jedoch gelungen w�re, sie
    f�r die Dauer unten zu halten, so tief sie auch liegen mochten, und
    zu verh�ten, da� sie nach vielen Jahren ans Tageslicht kamen und die
    M�rder durch ihren Anblick so sehr um alle Besinnung brachten, da�
    sie ihre Schuld eingestanden und am Galgen ihr Leben endeten. Ferner
    las er hier von Menschen, die in der Stille der Nacht in ihrem Bette
    liegend von ihren eigenen b�sen Gedanken zu so gr��lichen Mordtaten,
    wie sie selbst sagten, aufgestachelt wurden, da� es einen kalt �berlief
    und einem die Glieder matt am Leibe niedersanken, wenn man es las. Die
    f�rchterlichen Beschreibungen waren so lebensgetreu und packend, da�
    die schmutzigen Seiten ihm mit Blut bespritzt erschienen und die Worte,
    die er las, in seinen Ohren widerhallten, als w�rden sie in hohlem
    Murmeln von den Geistern der Ermordeten gefl�stert.

    In wahnsinniger Angst schlo� Oliver endlich das Buch und schleuderte
    es von sich, fiel auf die Knie nieder und flehte den Himmel an, ihn
    vor solchen Untaten zu bewahren und ihn lieber sogleich sterben als
    so f�rchterliche Verbrechen begehen zu lassen. Er wurde allm�hlich
    ruhiger und betete mit leiser, gebrochener Stimme um Errettung aus
    den Gefahren, in welchen er sich befand, und, falls einem armen,
    versto�enen Knaben, der nie Elternliebe und Schutz gekannt, Beistand
    und Hilfe aufgehoben w�re, da� sie ihm jetzt zuteil werden m�chte, wo
    er allein und verlassen von Schuld und Ruchlosigkeit umringt war.

    Er lag noch, das Gesicht mit den H�nden bedeckend, auf den Knien, als
    ein Ger�usch ihn aufschreckte. Er sah sich um, erblickte eine Gestalt
    an der T�r und rief: �Wer ist da?�

    �Ich -- ich bin es�, erwiderte eine bebende Stimme.

    Er hob das Licht empor und erkannte Nancy.

    �Stell das Licht wieder auf den Tisch�, sagte sie, das Gesicht
    abwendend; �die Augen tun mir weh davon.�

    Oliver sah, da� sie sehr bla� war, und fragte sie mitleidig, ob sie
    krank w�re. Sie warf sich auf einen Stuhl, so da� sie ihm den R�cken
    zukehrte, und rang die H�nde, antwortete aber nicht.

    �Gott verzeih' mir die S�nde!� rief sie nach einiger Zeit aus; �es ist
    meine Absicht nicht gewesen -- ich habe nicht -- habe nicht von fern
    daran gedacht!�

    �Ist ein Ungl�ck vorgefallen?� fragte Oliver. �Kann ich dir helfen?
    Wenn ich es kann, so will ich's auch, gern, gern!�

    Sie wiegte sich unter fortw�hrendem heftigen H�nderingen hin und her,
    fa�te sich an die Kehle, als ob sie etwas w�rgte, und keuchte atemlos.

    �Nancy!� rief Oliver best�rzt; �was ist dir denn?�

    Sie schlug krampfhaft mit den H�nden auf ihre Knie und mit den F��en
    auf den Boden und h�llte sich darauf schaudernd dicht in ihren Schal.
    Oliver sch�rte das Feuer an, sie setzte sich an den Kamin, schwieg noch
    eine Zeitlang, hob endlich den Kopf empor und blickte umher.

    �Ich wei� nicht, wie mir bisweilen wird�, sagte sie und stellte sich,
    als wenn sie eifrig besch�ftigt w�re, ihr Haar wieder zu ordnen; �ich
    glaube, jetzt kommt's von der dumpfen Luft hier im Zimmer. Bist du
    bereit, mit mir zu gehen, Nolly?�

    �Soll ich mit dir fortgehen, Nancy?� fragte Oliver.

    �Ja; ich komme von Bill Sikes�, erwiderte sie. �Du sollst mit mir
    gehen.�

    �Wozu denn?� fragte Oliver zur�ckschreckend.

    �Wozu?� wiederholte sie, schlug die Augen auf und wandte das Gesicht
    ab, sobald sie Olivers Blicken begegneten. �Oh! zu nichts B�sem.�

    �Das glaube ich dir nicht�, sagte er. Er hatte sie genau beobachtet.

    �So ist's gelogen, und glaub', was du willst�, erwiderte sie und zwang
    sich zu lachen. �Zu nichts Gutem also.�

    Oliver entging es nicht, da� er einige Gewalt �ber Nancys bessere
    Gef�hle hatte, und wollte sich schon an ihr Mitleid mit seiner
    hilflosen Lage wenden; allein es fiel ihm ein, da� es kaum elf Uhr
    w�re, da� noch viele Leute auf den Stra�en sein m��ten, und da� ihm
    ja wohl der eine oder andere Glauben schenken w�rde, wenn er ihn um
    Beistand anspr�che. Er trat vor, als ihm dieser Gedanke durch den Sinn
    flog, und erkl�rte hastig und verwirrt, da� er bereit sei.

    Nancy hatte ihn indes scharf im Auge behalten, erraten, was in seinem
    Innern vorging, und ihr bedeutsamer Blick lie� ihn gewahren, da� sie
    ihn durchschaut.

    �Pst!� sagte sie, beugte sich herunter zu ihm, blickte vorsichtig
    umher und wies nach der T�r. �Du kannst dir nicht helfen. Ich habe
    mir deinetwegen alle m�gliche M�he gegeben, aber vergeblich. Du bist
    umstellt und wirst scharf bewacht, und kannst du jemals loskommen, so
    ist es jetzt nicht die Zeit.�

    Sie war offenbar erregt, Oliver war davon betroffen und blickte ihr
    sehr verwundert in das Gesicht. Sie schien die Wahrheit zu reden, war
    bla� und zitterte heftig.

    �Ich habe dich schon einmal vor �bler Behandlung gesch�tzt, will es
    auch k�nftig tun und tue es jetzt�, fuhr sie fort; �denn wenn ich
    dich nicht holte, w�rden dich andere zu Sikes bringen, die viel
    unglimpflicher mit dir umgehen m�chten. Ich habe mich daf�r verb�rgt,
    da� du ruhig und still sein w�rdest, und bist du es nicht, so wirst du
    nur dir selbst und obendrein mir schaden, vielleicht an meinem Tode
    schuld sein. Sieh hier! -- dies alles hab' ich f�r dich schon ertragen,
    so wahr Gott sieht, da� ich's dir zeige.�

    Sie wies ihm mehrere braune und blaue Streifen und Flecke an ihrer
    Schulter und den Armen und sprach rasch weiter: �Denk daran, und la�
    mich nicht eben jetzt noch mehr um deinetwillen leiden. Wenn ich dir
    helfen k�nnte, w�rde ich's gern tun, ich habe aber die Macht nicht. Sie
    wollen dir kein Leides zuf�gen, und was sie dich zwingen zu tun, ist
    nicht deine Schuld. Pst! jedes Wort, was du sprichst, ist soviel als
    ein Schlag f�r mich. Gib mir die Hand -- geschwind, deine Hand!�

    Oliver reichte ihr mechanisch die Rechte, sie blies das Licht aus und
    zog ihn nach. Die Haust�r wurde rasch und leise von jemand ge�ffnet und
    ebenso schnell hinter ihnen wieder verschlossen. Vor dem Hause stand
    ein Mietswagen, sie schob ihn hinein und lie� die Fenster herunter.
    Der Kutscher bedurfte keiner Weisung, sondern fuhr augenblicklich im
    raschesten Trabe davon.

    Nancy hielt fortw�hrend Olivers Hand fest und fl�sterte ihm Trost,
    Warnungen und Versprechungen in das Ohr. Alles war so �berraschend
    gekommen, da� er kaum Zeit hatte, seine Gedanken zu sammeln, als der
    Wagen schon vor dem Hause hielt, in welchem der Jude am vergangenen
    Abend Sikes aufgesucht hatte.

    Einen einzigen kurzen Augenblick schaute Oliver umher, und ein
    Hilferuf schwebte ihm auf den Lippen. Allein die Stra�e war �de und
    menschenleer. Nancys bittende Stimme t�nte in seinem Ohr, und w�hrend
    er noch unentschlossen war, befand er sich schon im Hause und h�rte
    dasselbe sorgf�ltig verriegeln. Sikes trat mit einem Lichte oben an die
    Treppe und begr��te das M�dchen ungew�hnlich heiter und mild.

    �Tyras ist mit Tom nach Hause gegangen�, sagte er; �er w�rde im Wege
    gewesen sein.�

    �Das ist sch�n�, erwiderte Nancy.

    �Du hast ihn also?� bemerkte Sikes, als sie in das Zimmer eintraten.

    �Ja, hier ist er.�

    �Ging er ruhig mit?�

    �Wie ein Lamm.�

    �Freue mich, es zu h�ren,� sagte Sikes, Oliver finster anblickend,
    �um seines jungen Leichnams willen. Komm her, Bursch, da� ich dir nur
    gleich 'ne gute Lehre gebe, je eher, desto besser.�

    Er setzte sich an den Tisch, und Oliver mu�te sich ihm gegen�ber
    hinstellen.

    �Wei�t du, was dies ist?� fragte er, eine Taschenpistole zur Hand
    nehmend.

    Oliver bejahte.

    �Dann schau hier. Dies ist Pulver, dies 'ne Kugel und das ein
    Pfropfen.� -- Sikes lud die Pistole mit gro�er Sorgfalt und sagte, als
    er fertig war: �Nun ist sie geladen.�

    �Ja, Sir, ich sehe es�, erwiderte Oliver bebend.

    Sikes fa�te die Hand des Knaben mit festem Griffe und setzte ihm den
    Pistolenlauf an die Schl�fe. Oliver konnte einen Angstschrei nicht
    unterdr�cken.

    �Nun merk wohl, Bursch,� sagte Sikes, �sprichst du ein einziges Wort,
    wenn du mit mir au�er dem Hause bist, ausgenommen um zu antworten,
    wenn ich dich frage, so hast du ohne weiteres die ganze Ladung im
    Hirnkasten; also wenn du gesonnen sein solltest, ohne Erlaubnis zu
    sprechen, so sag erst dein letztes Gebet her. Soviel ich wei�, wird
    niemand besondere Nachforschung deinethalben anstellen, wenn dir der
    Garaus gemacht ist; 's ist also blo� zu deinem Besten, da� ich mir so
    viel M�he gebe, dir � Licht aufzustecken. Hast's geh�rt?�

    Jetzt nahm Nancy das Wort und sagte sehr nachdr�cklich und Oliver
    etwas finster anblickend, wie um ihn aufzufordern, ihr so aufmerksam
    als m�glich zuzuh�ren: �Das Lange und Kurze von dem, was du sagen
    willst, ist dies, Bill: Wenn er dich st�rt bei dem, was du vorhast, so
    wirst du ihm, damit er nichts ausschwatzen kann, eine Kugel durch den
    Kopf schie�en und die Gefahr auf dich nehmen, daf�r zu baumeln, wie du
    diese Gefahr wegen sehr vieler anderer Dinge auf dich nimmst, die du im
    Gesch�ft jede Woche deines Lebens tust.�

    �Ganz recht!� bemerkte Sikes wohlgef�llig. �Die Weibsen verstehen sich
    drauf, alles mit den wenigsten Worten zu sagen, ausgenommen, wenn sie
    zanken und schimpfen, wo sie's desto l�nger machen und die Worte nicht
    sparen. Jetzo aber, nun er Bescheid wei�, schaff was zum Abendessen,
    und dann wollen wir noch � bissel dormen[AC], eh' wir losgehen.�

      [AC] schlafen.

    Nancy gehorchte, deckte den Tisch und verschwand auf ein paar Minuten;
    dann kehrte sie mit einem Krug Porter und einer Sch�ssel Hammelfleisch
    zur�ck, die sie auf den Tisch stellte. Sikes a� und trank t�chtig und
    warf sich auf das Bett, nachdem er Nancy geboten, ihn Punkt f�nf Uhr zu
    wecken, und Oliver, sich auf die Matratze neben seinem Bette zu legen.
    Nancy sch�rte das Feuer und setzte sich an den Kamin, um die bestimmte
    Zeit nicht zu verfehlen.

    Oliver wachte noch lange und meinte, da� Nancy ihm vielleicht noch
    ein paar Worte zufl�stern w�rde; allein sie regte sich nicht, und er
    schlief endlich ein.

    Als er erwachte, stand Teegeschirr auf dem Tische; Sikes steckte
    verschiedene Sachen in die Taschen seines �ber einer Stuhllehne
    h�ngenden �berrocks, und Nancy war besch�ftigt, das Fr�hst�ck zu
    bereiten. Der Tag war noch nicht angebrochen, und das Licht brannte
    noch; der Regen schlug gegen die Fenster, und der Himmel sah schwarz
    und wolkig aus.

    Sikes trieb Oliver zur Eile an, der hastig sein Fr�hst�ck einnahm,
    worauf ihm Nancy ein Halstuch umband; Sikes hing ihm einen gro�en,
    groben Mantel �ber die Schultern, fa�te ihn bei der Hand, zeigte ihm
    den Kolben der Pistole und ging mit ihm fort, nachdem er sich von Nancy
    verabschiedet hatte.

    Oliver drehte sich an der T�r um, in der Hoffnung, einen Blick von
    Nancy zu erhalten, die sich jedoch schon wieder an den Kamin gesetzt
    hatte und regungslos in das Feuer schaute.




    21. Kapitel.

        Der Aufbruch.


    Es war ein unfreundlicher Morgen, als sie auf die Stra�e hinaustraten.
    Es ging ein scharfer Wind, und es regnete stark. Auf der Stra�e standen
    gro�e Pf�tzen, und die Rinnsteine waren �berf�llt. Am Himmel zeigte
    sich ein schwacher Schimmer des kommenden Tages, der aber das D�stere
    der Szene eher verst�rkte als verminderte, da das tr�be Licht nur dazu
    diente, das der Stra�enlaternen zu d�mpfen, ohne einen w�rmeren oder
    lichteren Farbenton in das Grau der nassen D�cher und schmutzigen
    Stra�en zu bringen. Es schien noch niemand in diesem Stadtviertel
    aufgestanden zu sein; die Fensterl�den der H�user waren noch fest
    verschlossen, und niemand lie� sich auf den �den, schmutzigen Stra�en
    blicken.

    Der Tag brach erst an, als sie sich Bethnal Green n�herten. Viele
    Laternen waren schon gel�scht; dann und wann fuhr ein Marktwagen
    langsam daher, oder es rollte eine Postkutsche vor�ber. Die Schenken
    standen schon offen und waren hell erleuchtet. Allm�hlich begannen
    sich auch einige L�den zu �ffnen. Hier kamen Gruppen von Arbeitern,
    die zur Werkstatt oder Fabrik gingen, dort M�nner und Frauen
    mit Fischk�rben auf dem Kopfe, mit Gem�se beladene Eselkarren,
    Fleischerwagen mit geschlachtetem Vieh, Milchfrauen mit ihren Kannen
    -- ein ununterbrochener Menschenstrom, der zu Fu� oder zu Wagen in die
    �stlichen Vorst�dte hineinflutete. Als sie sich der City n�herten,
    wurde der L�rm und der Verkehr immer st�rker; als sie die Stra�en
    zwischen Shoreditch und Smithfield durchschritten, war er zu einem
    sinnbet�ubenden Gew�hl angeschwollen. In Shoreditch und Smithfield
    war lautes Get�mmel und Gedr�nge. Es war Markttag. Oliver war vor
    Erstaunen au�er sich. Er meinte, ganz London w�re aus einer ganz
    besonderen Veranlassung in Bewegung. Welch eine Gesch�ftigkeit, welch
    ein Gew�hl, Rufen, L�rmen, Zanken und Streiten -- jeden Augenblick neue
    Gegenst�nde, neue Gesichter und Menschenkn�uel.

    Sikes zog seinen Begleiter rastlos fort, beachtete kaum, was Oliver die
    Sinne verwirrte, nickte nur dann und wann einem begegnenden Bekannten
    einen Gru� zu und lenkte nach Holborn ein. Er trieb zur Eile an, Oliver
    vermochte, fast atemlos, kaum Schritt mit ihm zu halten und wurde so
    rasch fortgerissen, da� es ihm fast war, als wenn er �ber die Erde
    dahinschwebte. Auf der Stra�e nach Kensington hielt Sikes einen leeren
    Karren an und forderte den Eigent�mer auf, ihn selbst und seinen Knaben
    bis Isleworth mitzunehmen. Er war mit dem K�rrner bald einig und hob
    Oliver in den Karren, wobei er nicht verga�, bedeutsam auf seine
    Rocktasche zu schlagen.

    Nachdem sie durch Kensington, Hammersmith, Chiswick, Kew Bridge und
    Brentford gefahren waren, lie� Sikes halten, stieg mit Oliver aus,
    wartete, bis der Fuhrmann vollst�ndig aus seinem Gesichtskreise
    verschwunden war, und setzte dann mit Oliver seine Wanderung fort. Sie
    wandten sich erst nach links, dann nach rechts und kamen an vielen
    gro�en G�rten und sch�nen Villen vor�ber, kehrten aber die ganze
    Zeit �ber nur einmal ein, um einen Schluck Bier zu trinken. Endlich
    erreichten sie eine Stadt, und Oliver sah an der Wand eines Hauses mit
    gro�en Buchstaben den Namen �Hampton� geschrieben. Sie warteten einige
    Stunden zwischen den Feldern und wandten sich dann nach der Stadt
    zur�ck; Sikes kehrte in einem alten, verfallenen Wirtshause mit einem
    verblichenen Schilde ein und bestellte ein Mittagessen beim K�chenfeuer.

    Die K�che war ein alter, niedriger Raum mit einem gro�en Balken quer
    �ber die Decke und hochlehnigen B�nken in der N�he des Feuers, auf
    denen mehrere rauh aussehende M�nner rauchend und trinkend sa�en. Sie
    beachteten Oliver gar nicht und Sikes sehr wenig, und da letzterer
    ebenfalls wenig Notiz von ihnen nahm, so sa� er mit seinem kleinen
    Gef�hrten ganz allein in einer Ecke, ohne sich durch ihre Anwesenheit
    im geringsten st�ren zu lassen.

    Sie a�en etwas kaltes Fleisch und blieben so lange sitzen, w�hrend
    Sikes sich den Genu� von drei bis vier Pfeifen g�nnte, da� Oliver ganz
    sicher glaubte, sie w�rden heute nicht weitergehen. Da er von der
    weiten Wanderung erm�det und so fr�h aufgestanden war, so wurde er
    schl�frig und versank endlich, �berw�ltigt von den Strapazen und dem
    Tabakrauch, in tiefen Schlummer.

    Es war schon ganz dunkel, als er durch einen Rippensto�, den ihm Sikes
    versetzt hatte, geweckt wurde. Als er sich gen�gend ermuntert hatte, um
    aufrecht sitzen und sich umschauen zu k�nnen, sah er seinen w�rdigen
    Begleiter mit einem Fuhrmann, der ziemlich betrunken zu sein schien
    und nach Shepperton wollte, bei einem Glase Ale zusammensitzen und
    h�rte, wie er ihn fragte, ob er ihn und den Knaben mitnehmen wolle.
    Der Fuhrmann willigte ein, und als es Zeit zum Abfahren war, hob Sikes
    Oliver in den Wagen, der sich sofort in Bewegung setzte und in scharfem
    Trabe aus der Stadt rasselte.

    Der Abend war sehr dunkel. Ein dichter Nebel stieg von dem Flusse und
    dem moorigen Boden ringsherum auf und breitete sich �ber die nassen
    Felder aus. Dazu war es schneidend kalt; alles war d�ster und schwarz.
    Kein Wort wurde gesprochen; denn der Fuhrmann war schl�frig geworden,
    und Sikes befand sich nicht in der Stimmung, ein Gespr�ch mit ihm
    anzukn�pfen. Oliver sa� zusammengekauert in einer Ecke des offenen
    Wagens, von Unruhe und Angst gepeinigt, und die riesigen B�ume, die wie
    in wilder Freude �ber die Trostlosigkeit der Gegend ihre Zweige heftig
    hin und her bewegten, kamen ihm wie gespenstische Wesen vor.

    Als sie an der Kirche in Sunbury vor�berfuhren, schlug es sieben
    Uhr. Sie mochten noch zwei oder drei Meilen gefahren sein, als Sikes
    abstieg und, Oliver bei der Hand fassend, weiterging. Er kehrte, was
    der m�de Knabe erwartet hatte, in Shepperton nicht ein, sondern ging
    durch den Schlamm, die Finsternis und d�stere Gassen und �ber �de,
    offene Pl�tze weiter, bis sich die Lichter einer Stadt in geringer
    Entfernung zeigten. Sie gelangten an eine Br�cke, und Sikes lenkte in
    einen Uferweg ein. Oliver erschrak heftig; er glaubte, da� Sikes ihn an
    diesen einsamen Ort gebracht h�tte, um ihn zu ermorden. Er wollte sich
    schon niederwerfen, um verzweifelt f�r sein junges Leben zu k�mpfen,
    als sie vor einem einzelnen, verfallenen Hause standen. Licht war darin
    nicht sichtbar; es schien unbewohnt zu sein. Sikes trat leise an die
    T�r, legte die Hand auf den Griff, und beide standen auf dem dunklen
    Hausflur.




    22. Kapitel.

        Der Einbruch.


    �Wer da?� rief eine laute heisere Stimme.

    �Mach keinen solchen Hamore[AD]�, sagte Sikes, w�hrend er die T�r
    verriegelte. �� Chandel[AE], Toby!�

      [AD] L�rm.

      [AE] Licht.

    �Aha, mein guter Chawwer�, erwiderte dieselbe Stimme; �� Chandel,
    Barney, � Chandel! F�hr den Herrn 'nein, Barney; wach aber erst auf,
    wenn dir's recht ist.�

    Man h�rte, da� irgend etwas Gewichtiges nach jemand geworfen wurde und
    sodann auf die Erde fiel.

    �H�rst nicht?� rief dieselbe Stimme. �Da steht Bill Sikes drau�en im
    Dunkeln, und du dormst, als wenn du 'nen Schlaftrunk g'soffen h�tt'st
    und nichts St�rkeres. Wirst du jetzt munter, oder soll ich dich mit'm
    eisernen Leuchter wecken?�

    Endlich schl�rfte der Kellner im Hotel von Saffron Hill mit Licht
    heran und begr��te Sikes mit wirklicher oder erk�nstelter Freude.
    Sikes stie� Oliver voran in ein niedriges, d�steres Gemach mit
    einigen gebrechlichen St�hlen, einem Tische und einem sehr schlechten
    Bette, auf welchem ein Mann ausgestreckt und aus einer langen
    Tonpfeife rauchend lag. Er trug einen dunkelbraunen Rock mit gro�en
    Metallkn�pfen, ein orangefarbenes Halstuch, eine buntfarbige Weste und
    hellbraune Beinkleider. Mr. Crackit (denn er war es) hatte d�nnes,
    r�tliches, in Locken gedrehtes Haar, durch das er von Zeit zu Zeit
    mit schmutzigen, beringten Fingern hindurchfuhr. Er war etwas �ber
    Mittelgr��e, und seine Beine schienen ziemlich d�nn zu sein, wodurch
    indes keineswegs die Bewunderung und Zufriedenheit vermindert wurde,
    womit er oft genug seine hohen Stiefel be�ugelte. �Bill, geliebter
    Freund,� rief er Sikes entgegen, �ich freue mich, dich zu sehen.
    F�rchtete fast schon, da� du's aufgegeben h�tt'st, in welchem Fall
    ich's auf meine eigene Faust versucht haben w�rde. Was der Teufel!�
    setzte er, als er Oliver erblickte, erstaunt hinzu, richtete sich zum
    Sitzen empor und fragte, wer der Knabe w�re.

    �Nun, 's ist eben der Knabe�, erwiderte Sikes und setzte sich an den
    Kamin.

    �Einer von Fagin seinen�, bemerkte Barney grinsend.

    �Von Fagin, so?� rief Toby, nach Oliver hinblickend, aus. �Was f�r'n
    prachtvoller Junge er werden wird f�r die Taschen der alten Damen in
    Kirchen und Kapellen! Sein Ponum[AF] ist so gut wie 'n Kap'tal f�r ihn.�

      [AF] Mund, Gesicht.

    �So schweig doch still -- 's ist schon mehr als zuviel Schw�tzens
    davon�, unterbrach ihn Sikes ungeduldig und fl�sterte ihm etwas in
    das Ohr. Toby Crackit lachte ausgelassen und starrte Oliver lange
    verwundert an.

    �Gebt uns zu essen und zu trinken -- es wird uns Courage machen -- mir
    wenigstens�, sagte Sikes. �Setz dich ans Feuer, Bursch, und ruh dich
    aus, du gehst noch mit uns aus heute nacht, wenn auch eben nicht weit.�

    Oliver sah ihn in stummer und furchtsamer Verwunderung an, setzte
    sich ans Feuer und st�tzte, kaum wissend, was um ihn her und mit ihm
    vorging, den schmerzenden Kopf auf die H�nde. Der j�dische J�ngling
    trug Speisen und Getr�nk auf, und Toby und Sikes tranken auf ein
    gl�ckliches Schr�nken. Toby f�llte ein Glas, reichte es Oliver und
    forderte ihn auf, es auszutrinken. Der Knabe versicherte, nicht trinken
    zu k�nnen, und bat mit jammervollen Mienen, ihn damit zu verschonen.
    Toby lie� sich jedoch nicht abweisen.

    �Hinunter damit!� rief er. �Meinst du, ich w��te nicht, was dir gut
    ist? Bill, sag's ihm, da� er trinkt!�

    �Soll ich dich lehren, gehorsam zu sein?� sagte Sikes, die Hand in
    die Tasche steckend. �Gott verdamm' mich, wenn mir der Bube nicht
    mehr Beschwerde macht, als ein ganz Dutzend Baldowerer. Trink aus,
    Galgenstrick, oder --!�

    Erschreckt durch die drohenden Geb�rden der beiden M�nner, st�rzte
    Oliver hastig den Inhalt des Glases hinunter und wurde sofort von
    einem heftigen Husten befallen, wor�ber Toby und Barney in ein lautes
    Gel�chter ausbrachen und sogar der gr�mliche Sikes den Mund verzog.

    Nachdem Sikes seinen Hunger gestillt hatte (Oliver konnte au�er einer
    Rinde Brot, die er zu essen gezwungen wurde, nichts zu sich nehmen),
    legten sich die beiden M�nner zu einem kurzen Schlafe nieder. Oliver
    blieb auf seinem Stuhle am Feuer sitzen, und Barney streckte sich, in
    eine Decke geh�llt, dicht neben dem Kamine aus.

    Sie schliefen oder schienen zu schlafen, denn es regte sich niemand
    au�er Barney, der ein paarmal aufstand, um Kohlen in das Feuer zu
    werfen. Oliver verfiel in einen dumpfen Schlummer, der von schweren,
    �ngstlichen Tr�umen beunruhigt wurde, bis Toby aufsprang und erkl�rte,
    es sei halb zwei Uhr. Im n�chsten Augenblicke waren die beiden
    anderen auf den Beinen, und alle drei waren eifrig dabei, die n�tigen
    Vorbereitungen zu treffen. Die beiden Schr�nker zogen sogleich ihre
    �berr�cke an und verh�llten sich mit T�chern bis �ber den Mund. Barney
    f�llte eiligst ihre Taschen mit mehreren Gegenst�nden an, die er aus
    einem Schranke holte.

    �Barney, meine Lupperts[AG]�, sagte Toby Crackit.

      [AG] Pistolen.

    �Da sind sie. Ihr habt sie selbst geladen.�

    �Ja, ja. Die Wurmer[AH].�

      [AH] Bohrer.

    �Die hab' ich�, fiel Sikes ein.

    �Chlamones[AI], Drehbarsel, H�nenehres[AJ], nichts vergessen?� fragte
    Toby, ein kleines Brecheisen einsteckend.

      [AI] Diebesschl�ssel.

      [AJ] Laterne.

    �Alles da�, antwortete Sikes. �Barney, die grandige Makel. Es ist
    h�chste Zeit.�

    Barney reichte ihm und Toby gro�e Knotenst�cke und legte Oliver den
    Mantel um.

    �Jetzt also�, sagte Sikes, seine Hand ausstreckend.

    Oliver, der durch die ungewohnte Anstrengung, die schlechte Luft und
    das ihm aufgezwungene Getr�nk v�llig bet�ubt war, legte seine Hand
    mechanisch in die Sikes'.

    �Nimm seine andere Hand, Toby�, sagte Sikes. �Schau nach, Barney, ob
    alles sicher ist.�

    Der Kellner ging vor die T�r und kehrte mit der Meldung zur�ck, da�
    alles still sei. Die beiden Schr�nker eilten hinaus und zogen Oliver
    mit sich fort.

    Die Nacht war rabenschwarz und der Nebel so dicht, da� nach wenigen
    Minuten gro�e Tropfen an Olivers Augenbrauen hingen. Sie eilten im
    tiefsten Schweigen �ber die Br�cke und durch den n�chstgelegenen Ort
    und erreichten um zwei Uhr ein einzeln stehendes, von einer Mauer
    umgebenes Haus, die Toby Crackit sogleich erklomm. Sikes hob Oliver
    empor, und nach wenigen Augenblicken waren alle drei hin�ber. Sikes
    und Toby schlichen nach dem Hause und zogen den Knaben mit sich fort,
    dem die Sinne fast entschwanden, denn jetzt zum erstenmal tauchte der
    Gedanke in ihm auf, da� Sikes auf Raub, wo nicht auf Mord ausginge
    und ihn als Werkzeug dabei zu gebrauchen denke. Er schlug die H�nde
    zusammen, und seinen Lippen entfloh ein unwillk�rlicher Schrei des
    Entsetzens. Ihm schwindelte, kalter Schwei� stand auf seiner Stirn, er
    wankte und fiel auf die Knie nieder.

    �Steh auf!� fl�sterte Sikes und zog bebend vor Wut die Pistole aus der
    Tasche; �steh auf, oder ich schie�e dir den Br�gen aus'm Kopfe 'raus!�

    �Oh, um Gottes willen, lassen Sie mich gehen!� rief Oliver; �lassen
    Sie mich fortlaufen und hinter dem Zaune sterben. Ich will nie wieder
    nach London kommen -- nie, nie; haben Sie Barmherzigkeit mit mir und
    zwingen Sie mich nicht, zu stehlen. Um der Liebe der Engel willen, die
    im Himmel wohnen, haben Sie Erbarmen mit mir!�

    Sikes stie� einen f�rchterlichen Fluch aus und spannte den Hahn, Toby
    schob indes seine Hand zur Seite, hielt Oliver den Mund zu und zog ihn
    fort nach dem Hause.

    �Pst!� fl�sterte er; �das ist hier nichts. Ist's nicht anders und
    soll's sein, so sprich ein Wort, und ich schlag' ihn auf den Kopf,
    was ebensogut ist und kein Ger�usch macht. Hierher, Bill, brich den
    Fensterladen auf. Ich stehe daf�r, er hat jetzt Courage genug. Ich
    hab's g'sehn, da� �ltere als er in 'ner kalten Nacht 's Kanonenfieber
    auf 'ne Minute oder so was g'habt haben.�

    Sikes murmelte Verw�nschungen gegen Fagin, ihm Oliver zu einem solchen
    Unternehmen geschickt zu haben, setzte das Brecheisen an, und nach
    kurzer Zeit war der Fensterladen ge�ffnet. Das kleine Gitterfenster war
    f�nf bis sechs Fu� �ber der Erde im Hinterhause und geh�rte zu einem
    kleinen, zum Waschen oder Brauen bestimmten Gemach am unteren Ende des
    Hausflurs. Das Gitter war gleichfalls bald durchbrochen.

    �Jetzt h�r' und merk, du kleiner Teufelsbraten!� fl�sterte Sikes, zog
    eine Blendlaterne aus der Tasche und hielt sie Oliver gerade vor das
    Gesicht. �Ich stecke dich durch dies Fenster hier. Nimm diese Laterne,
    geh leise die Stufen gerade vor dir 'nauf �ber den Flur nach der
    Haust�r, mach' sie auf und la� uns ein. -- Ist die Waschhaust�r offen,
    Toby?�

    Toby antwortete, nachdem er hineingesehen hatte: �Sie steht weit offen,
    und sie lassen sie immer offen, da� der Hund, der hier sein Lager hat,
    im Hause 'rumspazieren kann. Ha, ha, ha! Wie h�bsch ihn Barney gestern
    abend weggelockt hat!�

    So leise Crackit gesprochen und gekichert hatte, befahl ihm doch Sikes
    in gebieterischem Tone, still zu schweigen und an das Werk zu gehen.
    Er setzte die Laterne auf die Erde, stellte sich unter das Fenster,
    die H�nde auf die Knie gest�tzt, mit dem Kopfe gegen die Wand, Sikes
    stieg auf den R�cken Tobys und hob Oliver durch das Fenster in das Haus
    hinein.

    �Nimm die Leuchte�, fl�sterte er ihm zu. �Siehst du die Stufen da vor
    dir?�

    Oliver keuchte, mehr tot als lebendig, ein mattes �Ja�. Sikes wies mit
    der Pistole nach der Haust�r hin und erinnerte ihn, da� er ihn bis zur
    Haust�r fortw�hrend in Schu�weite h�tte und ihn niederschie�en w�rde,
    wenn er sich verweilte oder auch nur einen Schritt zur Seite ginge.

    �'s ist in 'ner Minute geschehen�, fl�sterte er Oliver zu. �Sobald ich
    dich loslasse, tu, was dir gehei�en ist. Pst!�

    �Was ist denn?� fragte Toby.

    Sie horchten.

    �Nichts�, sagte Sikes und lie� Oliver los. �Jetzt vorw�rts!�

    Der Knabe hatte sich indes wieder einigerma�en gesammelt und den
    raschen und festen Entschlu� gefa�t, wenn es auch sein Tod w�re, den
    Versuch zu machen, auf dem Hausflur zur Seite zu springen und L�rm zu
    machen. Von diesem Gedanken erf�llt, ging er bebend vorw�rts.

    �Komm zur�ck!� schrie Sikes pl�tzlich laut; �zur�ck, zur�ck!�

    Erschreckt durch die pl�tzliche Unterbrechung der Totenstille und ein
    lautes Geschrei, lie� Oliver die Laterne fallen und stand still, ohne
    zu wissen, ob er vorw�rts gehen oder entfliehen sollte. Das Geschrei
    wiederholte sich, es zeigte sich ein Licht -- es war ihm, als s�he er
    best�rzte, halb angekleidete M�nner an der T�r -- es schwamm ihm vor
    den Augen -- ein Gewehr blitzte auf -- ein Donner traf sein Ohr -- er
    taumelte zur�ck. Sikes fa�te ihn sogleich beim Kragen, feuerte nach den
    zur�ckweichenden M�nnern und zog ihn durch das Fenster.

    �Dr�ck den Arm dichter an den Leib�, fl�sterte er, w�hrend er ihn
    durchzog. �Toby, ein Tuch! Sie haben ihn getroffen. Geschwind! H�ll'
    und Teufel, wie der Bursch blutet!�

    Oliver war sich dunkel bewu�t, da� der L�rm im Hause immer mehr zunahm
    und da� er rasch fortgetragen wurde. Das Ger�usch verlor sich in der
    Ferne, die Sinne entschwanden ihm g�nzlich, es war ihm, als wenn eine
    kalte Hand sein Herz umfa�te, es schlug, und er sah und h�rte nichts
    mehr.




    23. Kapitel.

        Welches das Wesentliche einer anmutigen Unterredung zwischen Mr.
        Bumble und einer Dame enth�lt und zugleich dartut, da� sogar ein
        Kirchspieldiener in einigen Punkten empf�nglich sein kann.


    Der Abend war bitter kalt, und ein heftiger, schneidender Wind
    trieb dichte Schneewirbel durch die Luft. Es war ein Abend f�r die
    Wohlbeh�bigen, beim lustigen, prasselnden Feuer Gott zu danken, da�
    sie daheim waren, und f�r die heimatlosen Elenden und Hungrigen, sich
    niederzulegen und zu sterben. Ach! viele solcher Ausw�rflinge der
    Gesellschaft schlie�en zu solchen Stunden die Augen auf unseren �den,
    verlassenen Stra�en, und sie k�nnen dieselben, was auch ihr Verbrechen
    gewesen sein mag, kaum in einer schlimmeren Welt wieder �ffnen.

    So sah es drau�en aus, als Mrs. Corney, die Vorsteherin des
    Armenhauses, in welchem Oliver Twist das Licht der Welt erblickt
    hatte, sich in ihrem kleinen Zimmer an ihren behaglichen Kamin setzte
    und wohlgef�llig ihren kleinen runden Teetisch �berblickte, und als
    sie gar von dem Tische nach der Feuerstelle hinsah, wo der denkbar
    kleinste aller Kessel ein leises Lied mit leiser Stimme sang, wuchs
    augenscheinlich ihre innere Befriedigung, und zwar in einem solchen
    Grade, da� Mrs. Corney l�chelte.

    �Ja,� sagte sie, indem sie ihren Arm auf den Tisch st�tzte und sinnend
    ins Feuer blickte, �ich bin �berzeugt, wir haben alle volle Ursache,
    dankbar zu sein. Volle Ursache, wenn wir es nur anerkennen wollten.�

    Sie sch�ttelte betr�bt den Kopf, als wenn sie die geistige Blindheit
    der Armen beklagte, die es nicht erkannten, und fing an, ihren Tee zu
    bereiten, indem sie mit ihrem silbernen L�ffel (Privateigentum!) tief
    in eine zinnerne Teeb�chse fuhr.

    Wie geringe Dinge das Gleichgewicht unserer schwachen Gem�ter st�ren
    k�nnen! Der schwarze Teetopf war sehr klein und leicht gef�llt, das
    Wasser lief �ber und verbrannte ein wenig ihre Hand.

    �Oh, �ber den verw�nschten Topf!� sagte sie, ihn hastig aus der Hand
    setzend. �Das kleine dumme Ding h�lt nur ein paar Tassen. Wem ist er
    n�tze -- ausgenommen einer armen, einsamen, verlassenen Frau, wie ich
    es bin! Ach, ach!�

    Bei diesen Worten sank die w�rdige Dame auf ihren Stuhl und dachte,
    abermals den Arm auf den Tisch gest�tzt, �ber ihr Geschick nach. Der
    kleine Topf und die einzelne Tasse hatten traurige Erinnerungen an
    Mr. Corney (der noch nicht l�nger als f�nfundzwanzig Jahre tot war)
    erweckt. Sie war davon ganz �berw�ltigt.

    �Ich bekomme niemals einen anderen,� sagte sie kummervoll und mi�mutig;
    �bekomme niemals einen anderen -- wie ihn!�

    Wir k�nnen nicht entscheiden, ob sich dieser Sto�seufzer auf ihren
    Seligen oder den Teetopf bezog, auf welchen zum wenigsten ihre Blicke
    gerichtet waren, und der also auch gemeint sein konnte. Sie hatte kaum
    die erste Tasse gekostet, als leise geklopft wurde.

    �Herein!� rief Mrs. Corney �rgerlich. �Sicher will eins der alten
    Weiber sterben. Sie sterben immer, wenn ich bei Tisch sitze oder meine
    Tasse Tee trinke. Bleiben Sie nicht da drau�en stehen; Sie lassen sonst
    die kalte Zugluft herein. Was ist denn schon wieder los?�

    �Nichts, Ma'am, nichts�, antwortete eine M�nnerstimme.

    �Himmel! sind Sie es wirklich, Mr. Bumble?� rief die Dame jetzt weit
    freundlicher aus.

    �Zu Diensten, Ma'am�, erwiderte Bumble, der drau�en stehengeblieben
    war, um seine Schuhe zu reinigen und den Schnee von seinem Hute zu
    sch�tteln, und der jetzt eintrat, in der einen Hand seinen dreieckigen
    Hut und in der anderen ein B�ndel. �Darf ich die T�r schlie�en, Ma'am?�

    Mrs. Corney z�gerte versch�mt, zu antworten, weil es als eine
    Ungeschicklichkeit angesehen werden konnte, wenn sie mit Mr. Bumble bei
    geschlossener T�r eine Unterredung unter vier Augen h�tte, und Bumble
    benutzte die Z�gerung, um die T�r ohne erhaltene Erlaubnis zu schlie�en.

    �Schlechtes Wetter, Mr. Bumble�, bemerkte die Matrone.

    �Ja, ja, Ma'am,� sagte Bumble, �schlechte Witterung f�r das Kirchspiel.
    Wir haben heute nachmittag zwanzig Brote und anderthalb K�se
    weggegeben, und das Armenpack ist doch nicht zufrieden. Da ist ein
    Mann, der in Anbetracht seiner Frau und einer zahlreichen Familie ein
    gro�es Brot und ein ganzes Pfund K�se erhielt, und bedankte er sich,
    bedankte er sich wohl? Prosit die Mahlzeit! Er bettelte obendrein
    um Kohlen, und wenn's auch nur ein Taschentuch voll w�re, sagte er.
    Und was wollte er mit den Kohlen? Seine K�se dar�ber r�sten und dann
    wiederkommen und um noch mehr betteln! So machen sie's, Ma'am -- so
    machen sie's alle. Geben Sie ihnen eine Sch�rze voll Kohlen, und sie
    werden �bermorgen wiederkommen und eine neue haben wollen -- die
    Frechdachse! Vorgestern kam ein Mann, der kaum einen Fetzen auf seinem
    Leibe hatte (hier schlug Mrs. Corney versch�mt die Augen nieder) --
    Sie sind verheiratet gewesen, Ma'am, und so kann ich's wohl sagen --
    in des Direktors Haus, als der Herr gerade eine Mittagsgesellschaft
    hatte, und bat um Unterst�tzung. Da er nicht fortgehen wollte und die
    Gesellschaft bel�stigte, lie� ihm der Direktor ein Pfund Kartoffeln
    und ein Ma� Hafermehl reichen. >Mein Gott,< sagte der undankbare
    B�sewicht, >was soll ich damit? Sie k�nnten mir ebensogut 'ne eiserne
    Brille geben.< -- >Sehr wohl,< erwiderte ihm der Direktor, die Spende
    wieder an sich nehmend. >Ihr werdet hier sonst nichts bekommen.< --
    >Dann sterb' ich auf der offenen Stra�e<, sagte der Landstreicher. >Das
    werdet Ihr wohl bleiben lassen<, sagte der Direktor. Der Bettler ging
    und starb auf der Stra�e. Was sagen Sie zu 'nem solchen Eigensinne,
    Mrs. Corney?�

    �Es �bersteigt alle Begriffe�, versetzte die Dame. �Aber halten Sie als
    ein Mann von Erfahrung die Unterst�tzungen au�erhalb des Armenhauses
    nicht f�r sehr nachteilig, Mr. Bumble?�

    �Mrs. Corney,� erwiderte der Kirchspieldiener mit dem L�cheln bewu�ter
    �berlegenheit, �es ruht vielmehr in ihnen des Kirchspiels Schutz
    und Sicherheit. Ihr gro�es Prinzipium ist, den Armen just das zu
    geben, dessen sie nicht bed�rfen; sie werden es dann �berdr�ssig,
    wiederzukommen. Deshalb, Mrs. Corney, ist in den impertinenten
    Zeitungen so oft die Rede davon, da� arme Kranke mit K�se unterst�tzt
    w�rden, was jetzt im ganzen Lande die Regel ist. Dies sind jedoch
    Dienstgeheimnisse, wovon zu reden jedermann verboten sein sollte,
    ausgenommen uns Kirchspielbeamten. Mrs. Corney,� f�gte Bumble, sein
    B�ndel �ffnend, hinzu, �dies ist echter Portwein von bester Qualit�t,
    den das Kollegium f�r die Kranken abzuziehen befohlen hat.�

    Er stellte die beiden mitgebrachten Flaschen auf die Kommode, steckte
    sein Tuch bed�chtig in die Tasche und schickte sich zum Fortgehen an.
    Die mitleidige Dame bemerkte, es w�re recht kaltes Wetter, und fragte
    ihn sch�chtern, ob ihm nicht beliebe, ein Sch�lchen Tee anzunehmen. Er
    legte sogleich den Hut wieder aus der Hand, nahm an dem kleinen, runden
    Tische Platz, l�chelte und blickte Mrs. Corney so z�rtlich an, da� sie
    verlegen wegsehen und den Teekessel anblicken mu�te. Sie schenkte ihm
    ein, er breitete sein Taschentuch �ber die Knie und fing an zu trinken
    und zu essen, seinen Genu� von Zeit zu Zeit mit einem tiefen Seufzer
    begleitend, was jedoch seinem Appetit keineswegs schadete, sondern
    denselben vielmehr zu st�rken schien.

    �Ich sehe, Ma'am,� sagte er nach ziemlich geraumer Zeit, �Sie haben
    eine Katze und auch kleine K�tzchen.�

    �Sie glauben gar nicht, wie lieb ich sie habe, und wie vergn�gt und
    lustig sie bei mir sind, Mr. Bumble.�

    �Mrs. Corney, ich mu� sagen: jede Katze, die bei Ihnen und t�glich um
    Sie w�re und Sie nicht lieb h�tte, m��te ein Esel sein.�

    �Ah, Mr. Bumble!�

    �Es ist die Wahrheit, und ich w�rde sie mit Vergn�gen ers�ufen.�

    �Mr. Bumble, was Sie f�r ein hartherziger Mann sind!�

    �Ein hartherziger Mann?� wiederholte Bumble mit einem z�rtlichen
    Seufzer, ergriff und dr�ckte Mrs. Corneys kleinen Finger, r�ckte ein
    wenig um den Tisch herum und r�ckte immer n�her, bis sein Stuhl dicht
    neben dem Stuhle Mrs. Corneys stand, die nicht fortr�cken konnte, weil
    sie sonst dem Kamin zu nahe gekommen sein w�rde, was zwischen den
    beiden Feuern die noch gef�hrlichere N�he war. Rechts konnten ihre
    Kleider Feuer fangen, links nur ihr Herz; rechts konnte sie auf den
    Rost, links nur in Mr. Bumbles Arme fallen. Sie war eine kluge und
    umsichtige Frau, berechnete ohne Zweifel die m�glichen Folgen, blieb
    ganz still sitzen und schenkte Mr. Bumble noch eine Tasse Tee ein.

    �Ein hartherziger Mann, Mrs. Corney?� sagte Bumble, seinen Tee
    umr�hrend und ihr in das Angesicht schauend; �sind Sie eine hartherzige
    Frau?�

    �Mein Gott! Was f�r eine Frage f�r einen unverheirateten Mann!� rief
    die Matrone aus. �Was wollen Sie damit sagen, Mr. Bumble?�

    Bumble trank bis auf den letzten Tropfen aus, verspeiste eine ger�stete
    Butterschnitte, entfernte die Krumen von seinen Knien, wischte sich
    die Lippen und k��te die Matrone bed�chtig.

    �Mr. Bumble!� rief die keusche Dame fl�sternd; denn ihr Schrecken war
    so gro�, da� ihr die Stimme fast versagte: �Mr. Bumble, ich werde
    schreien!�

    Bumble sagte gar nichts, sondern legte langsam und mit W�rde den Arm um
    ihren Leib. Da sie die Absicht, schreien zu wollen, bereits angek�ndigt
    hatte, so w�rde sie bei dieser neuen Keckheit nat�rlich geschrien
    haben; allein es wurde unn�tig, indem hastig an die T�r geklopft wurde,
    worauf Bumble ebenso eilig aufsprang und mit gro�er Vehemenz die
    Portweinflaschen abzust�uben anfing. Mrs. Corney rief: �Herein!� Eine
    alte Frau steckte den Kopf in das Zimmer und verk�ndete, da� die alte
    Sarah im Sterben l�ge.

    �Was geht es mich an!� entgegnete Mrs. Corney verdrie�lich. �Kann ich
    sie am Leben erhalten?�

    �Das kann freilich niemand, Ma'am; ihr ist nicht mehr zu helfen. Ich
    habe viel Kranke sterben sehen, kleine Kinder wie M�nner in ihren
    besten Jahren, und wei� es auf ein Haar, wann der Tod im Anzuge ist.
    Jedoch ist sie unruhig in ihrem Geist und sagt, da� sie Ihnen noch
    etwas Notwendiges anzuvertrauen h�tte. Sie k�nnte nicht ruhig sterben,
    eh' Sie nicht bei ihr gewesen w�ren, Ma'am.�

    Die w�rdige Matrone murmelte eine betr�chtliche Anzahl von
    Verw�nschungen gegen die alten Frauen, die niemals sterben k�nnten,
    ohne absichtlich ihre Vorgesetzten zu bel�stigen, h�llte sich in einen
    w�rmenden Mantel, bat Bumble, zu bleiben, bis sie wieder da w�re, und
    entfernte sich verdrie�lich und keifend mit der an sie abgeschickten
    alten Frau.

    Was Mr. Bumble tat, als er sich allein sah, war etwas unerkl�rlich. Er
    �ffnete n�mlich den Schrank, z�hlte die Teel�ffel, wog die Zuckerzange,
    pr�fte einen Milchgie�er, ob er auch von echtem Silber w�re, setzte,
    nachdem er seine Wi�begier befriedigt, den dreieckigen Hut auf und fing
    an, sehr gravit�tisch im Zimmer umherzutanzen, nahm darauf den Hut
    wieder ab, setzte sich an den Kamin, blickte umher und nahm offenbar im
    Geist ein Inventar �ber die im Zimmer befindlichen Mobilien auf.




    24. Kapitel.

        Welches sehr kurz ist, aber doch f�r wichtig befunden werden k�nnte.


    Die Alte, welche die Ruhe des Zimmers Mrs. Corneys gest�rt hatte, war
    keine unpassende Todesbotin. Die Jahre hatten ihren Leib gekr�mmt,
    alle ihre Glieder zitterten, denn sie war vom Schlage ger�hrt
    worden, und ihr runzliges, entstelltes Antlitz glich mehr einer
    grotesk-phantastischen Zeichnung als einem Werke aus den H�nden der
    Natur.

    Ach! wie wenige alte Gesichter gibt es, die uns durch ihre Sch�nheit
    erfreuen! Angst, Sorgen und K�mmernisse der Welt verwandeln das
    menschliche Antlitz, wie sie die Herzen umwandeln, und erst wenn jene
    schlummern und f�r immer vor�ber sind, schwinden die unruhig bewegten
    Wolken und verh�llen und verdunkeln den hellen Himmel nicht mehr. Es
    ist sehr h�ufig bei den Gesichtern der Toten der Fall, da� sie selbst
    in ihrer Erstarrung den l�ngst vergessenen Ausdruck schlummernder
    Kinder wieder annehmen und die Z�ge der Kinderjahre wieder bekommen,
    so ruhig und friedlich wieder werden, da� diejenigen, die sie in ihrer
    Kindheit gekannt, mit Ehrfurchtsschauern an ihren S�rgen niederknien
    und den Engel schon auf Erden schauen.

    Die Alte humpelte ihrer keifenden Vorgesetzten voran, blieb endlich
    keuchend stehen, um Atem zu sch�pfen, und Mrs. Corney nahm ihr das
    Licht aus der Hand und ging allein in das Zimmer der Sterbenden, in
    welchem eine Lampe d�ster brannte. Am Krankenbette sa� eine andere alte
    Frau, und am Kamine stand der Lehrling des Apothekers und Doktors und
    schnitt einen Zahnstocher aus einem Federkiel.

    �Ein kalter Abend, Mrs. Corney�, bemerkte der junge Herr, als die Dame
    eintrat.

    �Sehr kalt, in der Tat, Sir�, erwiderte die Vorsteherin im h�flichsten
    Tone.

    �Sie sollten bessere Kohlen von Ihren Lieferanten verlangen�, sagte der
    Apothekerlehrling; �diese hier taugen absolut nichts f�r ein so kaltes
    Wetter.�

    �Das ist Sache des Kollegiums, Sir�, erwiderte die Dame.

    Hier wurde das Gespr�ch durch das St�hnen der Kranken unterbrochen.

    �Oh,� sagte der junge Mann, indem er sein Gesicht dem Bette zugewandt,
    �mit der ist's vorbei.�

    �Wirklich?� fragte die Matrone.

    �Ich w�rde mich dar�ber wundern, wenn sie noch eine Stunde lebte. Heda,
    schl�ft sie, Alte?�

    Die W�rterin nickte. Der Lehrling machte Gebrauch von seinem
    Zahnstocher, w�hrend sich Mrs. Corney stumm an das Bett setzte, und
    schlich nach einigen Minuten auf den Zehen hinaus. Gleich darauf
    erschien auch die W�rterin wieder, die Mrs. Corney gerufen hatte,
    winkte der anderen, und beide setzten sich an den Kamin und fingen
    leise miteinander zu sprechen an.

    �Hat sie noch mehr gesagt, Anny, wie ich fort war?�

    �Kein Sterbensw�rtchen.�

    �Hat sie den gew�rmten Wein getrunken, den ihr der Doktor verordnete?�

    �Sie konnte keinen Tropfen hinunterbringen; ich trank ihn daher selbst
    aus, und er hat mir sehr gut geschmeckt.�

    �Ich wei� die Zeit noch sehr wohl, da sie's ebenso gemacht und
    hinterher weidlich dar�ber gelacht hat.�

    �Freilich; sie war 'ne lustige alte Seele, hat manch liebe Leiche
    angekleidet und so h�bsch ausstaffiert wie 'ne Wachspuppe. Ich hab' ihr
    mehr als hundertmal dabei geholfen.�

    Mrs. Corney hatte ungeduldig auf das Erwachen der Schlummernden
    gewartet, stand auf, trat zu den beiden alten Meg�ren und fragte
    �rgerlich, wie lange sie denn eigentlich warten sollte.

    �Nicht lange mehr, Mistre�. Wir brauchen nicht lange auf den Tod zu
    warten. Geduld, Geduld! er wird uns allen bald genug kommen.�

    �Halten Sie den Mund und sagen Sie mir, Martha, hat die Patientin
    fr�her auch schon so gelegen?�

    �Oft genug.�

    �Wird's aber nicht wieder tun�, fiel die andere W�rterin ein; �ich
    meine, sie wird nur noch einmal wieder aufwachen, und wohl zu merken,
    Mrs. Corney, nur auf eine kurze Zeit.�

    �Ob sie auf eine lange oder kurze Zeit erwacht, sie wird mich nicht
    hier finden. Ihr alle beide, bel�stigt mich nicht noch einmal um nichts
    und wieder nichts, sonst geht's euch schlecht. Ich habe durchaus nicht
    die Verpflichtung, alle alten Weiber im Hause sterben zu sehen, und was
    noch mehr sagen will, ich mag's und will's nicht. Merkt euch das, ihr
    unversch�mten alten Schlumpen! Habt ihr mich noch einmal zur N�rrin, so
    nehmt euch in acht, das sag' ich euch.�

    Sie ging hinaus, als ein Schrei der beiden W�rterinnen, die wieder an
    das Bett getreten waren, sie zum Stillstehen brachte. Die Kranke hatte
    sich kerzengerade emporgerichtet und streckte die Arme nach ihnen aus.
    �Wer ist da?� rief sie mit hohler Stimme.

    �Pst, pst! Legen Sie sich nieder�, sagte eine der W�rterinnen.

    �Ich lege mich lebendig nimmermehr, nimmermehr wieder nieder�, rief die
    Patientin. �Ich will mit ihr sprechen. Kommen Sie, Mrs. Corney, da� ich
    Ihnen ins Ohr fl�stern kann.�

    Sie fa�te die Vorsteherin beim Arme und dr�ckte sie auf einen Stuhl,
    der neben dem Bette stand, nieder und war im Begriff, zu sprechen,
    als sie bemerkte, da� die beiden W�rterinnen so nahe wie m�glich
    herangetreten waren, um zu horchen, und sagte mit matter Stimme:
    �Schicken Sie sie hinaus -- geschwind, o geschwind!�

    Mrs. Corney befahl ihnen, hinauszugehen, und die Sterbende fuhr fort:
    �H�ren Sie mich nun an! In diesem selbigen Zimmer -- diesem selbigen
    Bette lag einst eine h�bsche, junge Frau. Sie ward mit blutenden F��en,
    staub- und schmutzbedeckt ins Haus gebracht, wurde von einem Knaben
    entbunden und starb. Ich war ihre W�rterin. Ich will mich besinnen --
    in welchem Jahre war es doch?�

    �Auf das Jahr kommt's nicht an�, unterbrach Mrs. Corney ungeduldig.
    �Was haben Sie mir von ihr zu sagen?�

    �Was ich von ihr zu sagen habe -- oh, ich wei� es wohl�, murmelte
    die Sterbende, richtete sich pl�tzlich mit ger�tetem Gesicht und
    vorspringenden Augen wieder empor und schrie fast: �Ich bestahl sie!
    Sie war noch nicht kalt -- noch nicht kalt -- als ich's tat.�

    �Sie bestahlen sie? -- Um Gottes willen, was nahmen Sie ihr?�

    �Es -- das einzige, was sie hatte. Sie bedurfte Kleider, um sich vor
    der K�lte zu sch�tzen, und Speise, um nicht Hungers zu sterben, hatte
    es aber trotzdem aufbewahrt, trug es im Busen; und es war von Gold, und
    sie h�tte sich damit vom Tode erretten k�nnen.�

    �Gold! -- Weiter, weiter, Frau. Wer war die Mutter -- wann starb sie?�

    �Sie gab mir den Auftrag, es aufzubewahren, und vertraute mir als der
    einzigen Frau, die um sie war. Ich stahl es ihr schon in Gedanken,
    als sie's mir zeigte; und vielleicht bin ich auch am Tode des Kindes
    schuld! Man w�rde den Knaben besser behandelt haben, wenn man alles
    gewu�t h�tte.�

    �Alles gewu�t! -- Sprechen Sie, sprechen Sie!�

    �Der Knabe ward seiner Mutter so �hnlich, da� ich immer an sie denken
    mu�te, wenn ich ihn sah. Ach, die �rmste! -- und sie war so jung -- und
    so sanft und geduldig! Ich mu� Ihnen aber noch mehr sagen -- noch viel
    mehr; -- hab' ich's Ihnen noch nicht alles gesagt?�

    �Nein, nein, nein -- nur schnell -- oder es wird zu sp�t werden!�

    �Als die Mutter ihren Tod herannahen f�hlte, fl�sterte sie mir ins
    Ohr, wenn das Kind am Leben bliebe, so w�rde der Tag erscheinen, wo es
    sich beim Nennen des Namens seiner Mutter nicht beschimpft achten, und
    Freunde finden --�

    �Wie wurde das Kind getauft?�

    �Oliver. Das Gold, das ich stahl -- war --�

    �Was, ums Himmels willen, was war es?�

    Frau Corney beugte sich in h�chster Spannung �ber die Sterbende, die
    noch ein paar unverst�ndliche Worte murmelte und leblos auf das Kissen
    zur�cksank. --

    �Mausetot!� bemerkte eine der W�rterinnen, als Frau Corney die T�r
    wieder ge�ffnet hatte.

    �Und hatte gar nichts zu erz�hlen�, sagte Frau Corney und entfernte
    sich, als wenn nur etwas ganz Gew�hnliches vorgegangen w�re.




    25. Kapitel.

        Worin die Erz�hlung wieder zu Fagin und Konsorten zur�ckkehrt.


    W�hrend sich die erz�hlten Ereignisse im Armenhause zutrugen, kauerte
    Fagin br�tend an einem matten, rauchigen Feuer in seiner alten H�hle
    -- derselben, aus welcher Oliver von Nancy entfernt worden war. Er
    hielt einen Blasebalg auf seinen Knien, mit dem er sich augenscheinlich
    bem�hte, das Feuer zu hellerer Flamme anzufachen. Aber er war in
    tiefe Gedanken versunken und blickte unverwandt, die Ellbogen auf den
    Blasebalg gest�tzt und das Kinn auf seinen Daumen ruhen lassend, auf
    das rostige Gitter.

    An einem Tische hinter ihm sa�en der gepfefferte Baldowerer, Charley
    Bates und Tom Chitling bei einer Partie Whist. Der Baldowerer spielte
    mit dem Strohmanne und gewann fortw�hrend, die Karten mochten fallen,
    wie sie wollten. Chitling zahlte, sprach seine Verwunderung �ber
    Dawkins' stets gl�ckliches Spiel aus und erkl�rte, da� nicht gegen
    ihn �anzukommen� sei. Charley Bates lachte ausgelassen, und Fagin
    blickte auf und bemerkte, Tom m�sse sehr fr�h aufstehen, um gegen den
    Baldowerer zu gewinnen.

    �Ja, du mu�t fr�h aufstehen, wenn du das willst, Tom,� fiel Charley
    ein, �und obendrein die Stiefel �ber Nacht anbehalten und 'ne doppelte
    Brille aufsetzen.�

    Dawkins h�rte die ihm gezollten Lobspr�che mit philosophischem
    Gleichmute an, und zeichnete sinnig den Grundri� vom Newgategef�ngnis
    mit Kreide auf den Tisch.

    �Du bist grausam langweilig, Tommy�, sagte der Baldowerer nach einer
    Pause von mehreren Minuten. �Woran sollte er wohl denken, Fagin?�

    �Wie kann ich's wissen?� antwortete der Jude. �Vielleicht an seinen
    Verlust oder seinen angenehmen Aufenthalt auf dem Lande, woher er
    gekommen ist erst soeben. Ha, ha, ha! Ist's das?�

    �Falsch geraten�, fuhr der Baldowerer fort. �Was meinst du, Charley?�

    �Nun, ich meine,� erwiderte Master Bates grinsend, �da� er zuckers��
    gegen Betsy war. Schau, wie rot er wird! 's ist zum Totlachen -- Tommy
    verliebt! O Fagin, Fagin, welch ein Hauptspa�!�

    �La� ihn zufrieden�, sagte der Jude, Dawkins einen Wink gebend und
    Bates einen mi�billigenden Sto� mit dem Blasebalg versetzend. �Betsy
    ist 'ne schmucke Dirne. Mach dich immerhin an sie, Tom; mach dich
    immerhin an sie 'ran!�

    �Fagin,� nahm Chitling zornig das Wort, �das geht hier niemand was an.�

    �O nein�, erwiderte der Jude. �La� Charley doch schwatzen und lachen;
    er l��t's einmal nicht. Betsy ist 'ne artige Dirne. Tu, was sie dir
    sagt, Tom, und du wirst machen dein Gl�ck.�

    �Ich tue, was sie mir sagt,� fuhr Tom fort, �und w�re nicht in die
    Tretm�hle gesteckt worden, h�tt' ich ihren Rat nicht befolgt. Ihr habt
    aber am Ende 'nen guten Rebbes dabei gemacht -- nicht wahr, Fagin?
    Und was wollen sechs Wochen sagen? Es kommt doch einmal, fr�her oder
    sp�ter, und im Winter ist's just am besten, wenn einem nicht daran
    gelegen ist, so oft auszugehen -- he, Fagin?�

    �Sehr richtig, mein Lieber�, versetzte der Jude.

    �Es wird dir gewi� gleich viel ausmachen, Tom, noch einmal in die M�hle
    zu kommen,� fiel der Baldowerer, Fagin und Bates zublinzelnd, ein,
    �wenn nur alles mit Betsy in Richtigkeit w�re.�

    �Ja, das w�rd's -- seht!� erwiderte Tom noch erz�rnter, �und ich m�chte
    doch wissen, wer mir's nacht�te, Fagin?�

    �Das f�llt ein keiner Seele�, antwortete Fagin. �Ich wei� keinen au�er
    dir, der's w�rde tun.�

    �Ich h�tte ganz davonkommen k�nnen, h�tt' ich mosern wollen -- he,
    Fagin?� fuhr der halb bl�dsinnige Bursche, immer zorniger werdend,
    fort. �Ich h�tte nur ein einziges Wort zu sagen brauchen, nicht wahr,
    Fagin? Ich schwatzte aber nicht -- und was ist denn nun dabei zu
    lachen?�

    Fagin eilte, ihm zu versichern, da� niemand lache, nicht einmal Charley
    Bates, der jedoch, als er den Mund �ffnete, um auch seinerseits zu
    erkl�ren, da� alle ohne Ausnahme �u�erst ernsthaft gestimmt w�ren, in
    ein unbez�hmbares Gel�chter ausbrach. Tom Chitling sprang w�tend auf,
    um dem Frechen einen Schlag zu versetzen, allein Charley b�ckte sich
    gewandt, und der Schlag traf den munteren alten Herrn derma�en vor
    die Brust, da� derselbe gegen die Wand taumelte, und da� ihm der Atem
    verging.

    �Still! ich hab' den Bimbam g'h�rt�, rief der Baldowerer in diesem
    Augenblick, nahm das Licht vom Tische, schlich leise die Treppe hinauf,
    kehrte nach einer halben Minute zur�ck und fl�sterte Fagin etwas in das
    Ohr.

    �Wie?� rief der Jude. �Allein?�

    Der Baldowerer nickte und gab Charley Bates einen freundschaftlichen
    Wink, er t�te besser daran, seine Heiterkeit etwas zu z�geln. Dann
    blickte er wieder den Juden an und erwartete dessen Anweisungen.

    Der alte Mann bi� sich auf seine gelben Finger und sann einige
    Augenblicke nach. Sein Gesicht arbeitete w�hrenddessen heftig, als sei
    er erschrocken und f�rchte, das Schlimmste zu erfahren. Endlich erhob
    er den Kopf und fragte: �Wo ist er?�

    Der Baldowerer deutete nach oben und machte Miene, das Zimmer zu
    verlassen.

    �Ja,� sagte der Jude als Antwort auf diese stumme Frage, �bring ihn
    herunter. Pst, still, Charley und Tom, still, still!�

    Die Angeredeten gehorchten sofort. Sie gaben keinen Laut von sich, als
    der Baldowerer, das Licht in der Hand, die Treppe herabkam und ihm
    dicht auf den Fersen ein Mann folgte, der, nachdem er sich hastig im
    Zimmer umgeblickt hatte, ein gro�es Tuch abwarf, das bisher den unteren
    Teil seines Gesichts verdeckte, so da� die hageren, ungewaschenen und
    unrasierten Z�ge des blonden Toby zum Vorschein kamen. Er begr��te
    Fagin, der ihn �ngstlich fragend ansah, und erkl�rte sogleich, von
    Gesch�ften nicht eher reden zu k�nnen, als bis er gegessen und
    getrunken h�tte. Der Jude befahl Dawkins, aufzutragen, was vorhanden
    w�re; es geschah, und Toby machte sich begierig dar�ber her, ohne die
    mindeste Neigung zu zeigen, das Gespr�ch zu beginnen und der Ungeduld
    und Herzensangst des Juden ein Ende zu machen, der auf und ab laufend
    mit seinen Blicken jeden Bissen z�hlte und verw�nschte, den Toby zum
    Munde f�hrte. Toby l�chelte, w�hrend er speiste, selbstgef�llig und
    schmunzelnd wie immer, und der Jude h�tte vor Ingrimm vergehen m�gen.
    Endlich hub er an: �Vor allen Dingen, Fagin --�

    �Ja, ja doch -- vor allen Dingen --�

    �Vor allen Dingen, Fagin, wie steht's mit Bill?�

    �Wie -- mit Bill!� kreischte der Jude, vom Stuhle aufspringend, denn er
    hatte sich h�rbegierig dicht neben Toby gesetzt.

    �Zum Geier -- Ihr wollt doch nicht sagen --� fuhr Crackit erblassend
    fort.

    �Was soll ich nicht wollen sagen?� schrie der Jude, w�tend mit den
    F��en stampfend. �Wo sind sie? -- Sikes und der Knabe -- wo sind sie?
    -- wo sind sie geblieben? -- wo sind sie versteckt? -- warum sind sie
    nicht hier?�

    �Der Einbruch mi�gl�ckte�, erwiderte Toby mit unsicherer Stimme.

    �Ich wei� es�, sagte der Jude, ein Zeitungsblatt aus der Tasche nehmend
    und es Toby vorhaltend. �Was weiter?�

    �Es wurde geschossen und der Knabe getroffen. Wir machten uns mit ihm
    davon -- rannten und setzten �ber Hecken und Gr�ben, als wenn der
    Teufel selbst hinter uns w�re. Wir wurden verfolgt -- Gott verdamm'
    mich, die ganze Umgegend war lebendig, und wir hatten die Hunde auf den
    Fersen.�

    �Aber der Knabe, der Knabe!� keuchte Fagin.

    �Bill trug ihn auf dem R�cken; wir hielten an mit Laufen, um ihn
    zwischen uns zu nehmen; er lie� den Kopf h�ngen und war steif und
    kalt. Sie waren dicht hinter uns, und da galt's, jeder sich selbst der
    N�chste, wenn er nicht der erste am Galgen sein wollte. Wir rissen
    aus, der eine hier, der andere da hin, und lie�en den Burschen in 'nem
    Graben liegen -- ob tot oder lebendig, ich kann's nicht sagen. Das ist
    alles, was ich von ihm wei�.�

    Der Jude stie� einen gellenden Schrei aus, fuhr mit den H�nden in das
    Haar und st�rzte aus dem Zimmer und zum Hause hinaus.




    26. Kapitel.

        In welchem eine geheimnisvolle Person auftritt und viel von der
        Erz�hlung Untrennbares geschieht.


    Der alte Mann hatte die Stra�enecke erreicht, bevor er anfing, sich von
    dem Schrecken wieder zu erholen, den ihm Tobys Mitteilungen eingejagt
    hatten. Er eilte soviel wie m�glich durch Nebenstra�en und Gassen,
    fast sinnlos immer vorw�rts, so da� er beinahe von einem Mietswagen
    �berfahren worden w�re, und langte endlich auf Snow-Hill an, wo er
    seine Schritte noch beschleunigte, bis er in eine lange und enge Gasse
    eingebogen war. Jetzt schien er sich auf seinem Terrain zu f�hlen und
    freier zu atmen, denn er lief nicht mehr, sondern verfiel in seinen
    gew�hnlichen, halb trippelnden, halb schl�rfenden Gang.

    Nicht weit von der Stelle, wo Snow-Hill und Holborn-Hill
    zusammensto�en, �ffnet sich rechter Hand, wenn man aus der City
    kommt, eine nach Saffron-Hill f�hrende enge und erb�rmliche Stra�e
    -- Field-Lane -- mit zahllosen schmutzigen L�den, in welchen die
    Taschent�cher feilgeboten werden, welche die Ladenbesitzer von den
    Taschendieben erhandelt haben. Die Stra�e hat ihren eigenen Barbier,
    ihr Kaffeehaus, ihre Bierstube und ihre Gark�che. Sie bildet eine
    eigene Handelskolonie, ist der Stapelplatz f�r tausenderlei Artikel,
    die Industriefr�chte der kleineren Diebe, und wird am fr�hen Morgen
    und in der Abendd�mmerung von schweigsamen Handelsleuten besucht,
    die in finsteren Hinterzimmern ihre Gesch�fte abmachen und auf so
    absonderliche Art gehen, wie sie kommen.

    In Field-Lane lenkte der Jude ein. Er war den Bewohnern sehr wohl
    bekannt, von denen einer nach dem andern dem Vor�bergehenden
    vertraulich zunickte. Er erwiderte ihre Begr��ungen auf dieselbe Weise,
    hielt sich indes nirgends auf, bis er den Ausgang der Stra�e erreicht
    hatte, wo er einen Handelsmann von sehr kleiner Statur anredete, der in
    seinem Laden sa� und behaglich seine Pfeife rauchte. Er fragte ihn, wie
    er sich bef�nde.

    �Vortrefflich! Aber in aller Welt, Mr. Fagin, wie, bekommt man Euch
    einmal wieder zu sehen?� erwiderte das M�nnchen.

    �Die Nachbarschaft hier war zu hei� ein wenig, Lively!� sagte Fagin,
    die Augenbrauen emporziehend und die H�nde �ber der Brust kreuzend.

    �Hm! ich habe wohl schon ein paarmal dar�ber klagen h�ren; sie k�hlt
    sich indes bald wieder ab -- findet Ihr das nicht auch?�

    Fagin nickte, wies nach Saffron-Hill und fragte, ob dort zu Abend
    jemand w�re.

    �In den Kr�ppeln?� fragte der kleine Handelsmann.

    Der Jude bejahte.

    �Wartet mal�, fuhr der Handelsmann nachsinnend fort. �Ja, es ist ein
    halbes Dutzend hineingegangen, soviel ich gesehen habe. Ich glaube aber
    nicht, da� Euer Freund dort ist.�

    �Ist Sikes nicht da?� fragte Fagin mit der Miene get�uschter Erwartung.

    �Nein�, erwiderte der Kleine, mit einem unsagbar schlauen Ausdruck den
    Kopf sch�ttelnd. �Habt Ihr nichts zu handeln heute?�

    �Heute nicht�, erwiderte der Jude im Fortgehen.

    �Geht Ihr in die Kr�ppel, Fagin?� rief ihm der kleine Handelsmann nach.
    �Ich will mitgehen und 'nen Tropfen mit Euch trinken.�

    Fagin winkte ihm mit der Hand, ihm bedeutend, da� er allein zu bleiben
    w�nsche, und die Kr�ppel wurden somit f�r dieses Mal der Ehre des
    Besuchs Mr. Livelys beraubt, zumal der kleine Mann nicht leicht von
    seinem Gesch�ft abkommen konnte. W�hrend er sich erhoben hatte, war
    der Jude verschwunden, und nachdem Mr. Lively sich vergebens auf die
    Zehen gestellt hatte, um ihn nochmals zu Gesicht zu bekommen, mu�te er
    sich notgedrungen wieder auf seinen Stuhl setzen und nahm nach einem
    bedenklichen und mi�trauischen Kopfsch�tteln seine Pfeife wieder zur
    Hand.

    Die Kr�ppel waren das Gasthaus, in welchem Sikes und sein Hund bereits
    figuriert haben. Fagin gab einem Manne am Schenktische nur ein stummes
    Zeichen und ging geradeswegs die Treppe hinauf, �ffnete eine T�r, trat
    sacht hinein und blickte �ngstlich suchend und die Augen mit der Hand
    beschattend, umher.

    Das Zimmer war durch zwei Gasflammen erleuchtet, man hatte aber die
    Fensterl�den verschlossen und die Vorh�nge dicht zugezogen. Die
    Decke war geschw�rzt, damit ihre Farbe unter dem Qualm der Lampen
    nicht litte, und der ganze Raum dergestalt mit Tabaksrauch angef�llt,
    da� Fagin anfangs kaum einen Gegenstand zu unterscheiden vermochte.
    Allm�hlich erkannte er jedoch die zahlreiche Gesellschaft, deren
    Anwesenheit ihm zuerst nur durch verworrenen L�rm kund geworden war.
    Oben an der Tafel sa� mit einem Pr�sidentenhammer der Wirt, ein
    plumper, vierschr�tiger Mann, der, als ein munteres Lied gesungen
    wurde, sich g�nzlich der allgemeinen Heiterkeit hinzugeben schien,
    die Augen und Ohren aber -- und zwar sehr scharfe Augen und Ohren
    -- offen und �berall hatte. Ihm gegen�ber an einem verstimmten
    Fortepiano sa� ein Musiker mit bl�ulicher Nase und Zahnschmerzen
    halber verbundener Wange. Die S�nger lie�en sich ihre Gl�ser noch
    weit besser als die ihnen gespendeten Lobspr�che behagen, und die
    Gesichter ihrer Bewunderer dr�ckten fast jedes Laster in jeglicher
    Abstufung aus und waren unwiderstehlich anziehend, weil grenzenlos
    absto�end. Man sah �berall die mannigfachsten und wahrhaftesten
    Bilder der Verschmitztheit, Brutalit�t und Trunkenheit, und die --
    s�mtlich noch mehr oder minder jugendlichen -- Frauenzimmer trugen die
    abschreckendsten Spuren der Ausschweifung an sich, w�hrend in ihrem
    w�sten Aussehen keine Spur edler Weiblichkeit mehr zu entdecken war, so
    da� sie die schw�rzeste und betr�bendste Schattenpartie des Gem�ldes
    bildeten.

    Fagin lie� sich jedoch durch Gedanken solcher Art nicht von fern
    beunruhigen. Seine Blicke schweiften gespannt von einem Gesicht zum
    andern, schienen aber vergebens zu suchen. Er winkte endlich unbemerkt
    dem vorsitzenden Wirte, und schlich so sacht wieder hinaus, wie er
    hineingeschlichen war.

    �Was w�nscht Ihr von mir, Mr. Fagin?� fragte der Wirt leise, sobald er
    beim Juden drau�en an der Treppe stand. �Wollt Ihr Euch nicht zu uns
    setzen? Die ganze Gesellschaft w�rde sich sehr freuen.�

    Der Jude sch�ttelte ungeduldig den Kopf und fl�sterte: �Ist er hier?�

    �Nein.�

    �Keine Nachricht von Barney?�

    �Nein. Er wird sich auch nicht r�hren, bis alles sicher ist. Verla�t
    Euch drauf, sie sind ihm auf der Spur, und wenn er sich blicken lie�e,
    w�rde er die ganze Geschichte verraten. 's ist alles ganz richtig mit
    ihm: ich h�tte sonst von ihm geh�rt. La�t ihn nur zufrieden; ich stehe
    daf�r, da� er sich mit gro�er Klugheit benimmt.�

    �Wird er nicht kommen heut' abend?�

    �Meint Ihr Monks?� lautete des Wirtes z�gernde Gegenfrage.

    �Pst! Ja doch!�

    �Ich hab' ihn schon erwartet, und wenn Ihr nur zehn Minuten verweilen
    wollt --�

    �Nein, nein�, unterbrach ihn der Jude hastig, als ob es ihn beruhigt
    h�tte, zu h�ren, da� der Mann, nach welchem er gefragt, nicht anwesend
    sei, so begierig er, wie es schien, gewesen war, ihn zu sehen. �Sagt
    ihm, da� ich ihn gesucht h�tte hier, und da� er noch heute abend m��te
    kommen zu mir -- doch nein, sagt morgen. Da er einmal nicht hier ist,
    wird's auch morgen noch sein Zeit genug.�

    �Gut! Habt Ihr noch ein Anliegen?�

    �Nein, gute Nacht!� erwiderte Fagin im Hinuntergehen.

    �Holla!� rief ihm der Wirt fl�sternd nach, �was dies f�r 'ne
    Gelegenheit zu 'nem Gesch�ftchen sein w�rde! Ich hab' da den Phil
    Barker drinnen so sternig[AK], da� ihn ein Kind brennen[AL] k�nnte.�

      [AK] betrunken.

      [AL] betr�gen.

    �Ah so! 's ist aber noch nicht f�r Phil Barker die Zeit�, rief der Jude
    ebenso leise zur�ck. �Phil hat noch zu tun etwas, bis wir k�nnen ihn
    entbehren. Geht also wieder zu Eurer Gesellschaft, mein Lieber, und
    sagt den Leuten, da� sie lustig m�chten leben -- solange sie noch am
    Leben sind. Ha, ha, ha!�

    Der Wirt stimmte in das heisere Lachen des alten Mannes ein und kehrte
    zu seinen G�sten zur�ck. Sobald der Jude allein war, wurden auch seine
    Mienen wieder nachdenklich und besorgt. Nach einem kurzen Besinnen rief
    er einen Mietskutscher an, befahl ihm, nach Bethnal Green zu fahren,
    stieg einige tausend Schritte vor Sikes' Wohnung wieder aus und eilte
    zu Fu� weiter.

    �Jetzt wird sich's schon zeigen, mein M�dchen,� murmelte er vor sich
    hin, als er an die Haust�r klopfte; �f�hrst du was Geheimes im Schilde,
    so will ich's bald haben heraus, so listig du auch bist.�

    Er schlich leise hinauf und trat, ohne anzuklopfen, in Nancys Zimmer.
    Sie war allein und lag mit dem Kopfe, um den das Haar unordentlich
    herumhing, auf dem Tische. �Sie hat getrunken�, dachte er gleichg�ltig,
    �oder ist vielleicht blo� unwirsch.�

    Der alte Mann dr�ckte die T�r wieder zu, w�hrend er diese Betrachtung
    anstellte, und das dadurch hervorgebrachte Ger�usch weckte sie aus
    ihrem Schlummer oder Hinbr�ten; sie begegnete ruhig seinen forschenden
    Blicken, fragte, was es Neues g�be, und er erz�hlte ihr, was er von
    Toby Crackit vernommen hatte. Sie h�rte ihm zu, legte, ohne ein Wort
    zu sprechen, den Kopf wieder auf den Tisch, stie� dann das Licht
    ungeduldig von sich und scharrte mit den F��en; dies war jedoch alles.

    Der Jude blickte unruhig umher, als ob er sich �berzeugen wollte, da�
    Sikes nicht insgeheim zur�ckgekehrt w�re. Befriedigt, wie es schien,
    durch sein Umhersp�hen, hustete er ein paarmal und machte ebensoviele
    Versuche, ein Gespr�ch anzukn�pfen; allein das M�dchen beachtete ihn
    nicht mehr, als wenn er eine Bilds�ule gewesen w�re. Endlich nahm er
    sich zusammen und sagte h�ndereibend und im freundlichsten Tone: �Was
    meinst du denn, liebes Kind, wo wohl sein mag Bill?�

    Das M�dchen murmelte in kaum verst�ndlichen Worten, sie k�nne es nicht
    sagen, und es schien ihm, als ob sie leise schluchze.

    �Und wo wohl mag sein der kleine Oliver?� fuhr er fort, die Augen
    anstrengend, um etwas von ihrem Gesichte zu ersp�hen. �Das arme Kind --
    denk nur, Nancy -- wie sie's haben lassen liegen in einem Graben!�

    �Da ist ihm wohler als unter uns�, sagte das M�dchen, pl�tzlich
    aufblickend; �und wenn f�r Bill nichts Schlimmes daraus entsteht, so
    will ich hoffen und w�nschen, da� der Kleine tot im Graben liegt, und
    da� seine jungen Gebeine darin verfaulen.�

    Den Lippen des Juden entfloh ein Ausruf des Erstaunens.

    �Ja, das hoff' und w�nsch' ich�, fuhr Nancy, seinen Blicken begegnend,
    fort. �Ich freue mich, da� er mir aus den Augen, und zu wissen, da�
    das Schlimmste vor�ber ist. Ich *kann* ihn nicht um mich haben; ich
    verabscheue mich selbst und euch alle, wenn ich ihn sehe.�

    �Pah!� fiel der Jude ver�chtlich ein. �Du bist betrunken, M�dchen.�

    �So -- betrunken!� h�hnte Nancy. �Eure Schuld ist's freilich nicht,
    wenn ich's nicht bin. Ich w�re niemals n�chtern, wenn's nach Eurem
    Willen ginge, jetzt ausgenommen! -- Meine Laune scheint Euch nicht zu
    behagen.�

    �Nein, durchaus nicht!� sagte der Jude w�tend.

    �So �ndert sie�, fuhr das M�dchen mit Lachen fort.

    �Sie �ndern!� schrie der Jude, durch die unerwartete Hartn�ckigkeit
    des M�dchens und die Verdrie�lichkeiten des Abends �ber alle Ma�en
    erbittert. �Ja, ich will sie �ndern! H�r', was ich werde dir sagen,
    du liederliches Weibsbild! Ich, der ich nur zu sprechen brauche sechs
    Worte, und Sikes wird zugeschn�rt die Kehle so gewi�, wie ich w�rde
    ihn d�mpfen, h�tt' ich jetzt zwischen meinen Fingern seinen Stierhals.
    Kommt er zur�ck, ohne mitzubringen den Knaben -- kommt er gl�cklich
    davon und bringt mir nicht ihn, lebendig oder tot, M�dchen, so morde
    deinen Bill selbst, wenn du willst, da� er entgehen soll dem Galgen,
    und tu' es ja, sobald er den Fu� hier setzt hinein ins Zimmer; denn
    merk', es wird sonst sein zu sp�t!�

    �Was sagt Ihr da?� rief das M�dchen unwillk�rlich aus.

    �Was ich sage?� fuhr der Jude, vor Wut fast von Sinnen, fort. �Dies
    sag' ich! Wenn das Kind ist wert viele hundert Pfund f�r mich, soll ich
    verlieren, was mir zugew�rfelt hat der Zufall, durch die Tollheiten
    einer betrunkenen Bande, deren Leben in meiner Gewalt ist -- und indem
    ich obenein gesellt bin mit 'nem eingefleischten Teufel, der nur
    braucht zu wollen und hat die Macht, zu ... zu ...� --

    Er keuchte atemlos, sprudelte vor Wut, bem�hte sich vergebens, Worte
    zu finden; pl�tzlich aber bezwang er seinen Zorn und nahm ein ganz
    anderes Wesen an. Er sank zusammengekr�mmt auf einen Stuhl nieder und
    bebte vor Angst, geheimste Schurkereien selbst offenbart zu haben. Nach
    einem kurzen Stillschweigen wagte er es, nach Nancy hinzublicken und
    schien etwas ruhiger zu werden, als er sie wieder in derselben achtlos
    gleichg�ltigen Stellung sah, in welcher er sie gefunden hatte.

    �Nancy, liebes Kind,� kr�chzte er in seinem gew�hnlichen Tone, �hast du
    geh�rt, was ich habe gesagt?�

    �La�t mich jetzt in Ruhe, Fagin�, antwortete sie, den Kopf matt und
    schl�frig emporrichtend. �Wenn es Bill diesmal nicht getan hat, so
    wird er's ein andermal tun; er hat manch sch�nes Gesch�ft f�r Euch
    ausgerichtet und wird Euch noch viele ausrichten, wenn er kann; kann
    er's aber einmal nicht, so kann er's nicht. Und nun sprecht nicht mehr
    davon.�

    �Aber was anbelangt den Oliver, Kind?� sagte der Jude, indem er sich
    unruhig die H�nde rieb.

    �Er mu� das Schicksal der anderen teilen,� fiel Nancy hastig ein; �und
    ich sag' es noch einmal, ich hoffe, da� er tot ist und vor Schaden und
    vor Euch sicher ist -- das hei�t, wenn Bill nichts Schlimmes begegnet;
    und ist Toby gut davongekommen, so wird er's ohne Zweifel auch sein,
    denn was der kann, kann Bill tausendmal.�

    �Und was anbelangt das, was ich sagte, Kind?� sagte der Jude, sie
    doppelt scharf in das Auge fassend.

    �Ihr m��t's alles noch einmal wiederholen, wenn Ihr wollt, da� ich
    etwas tun soll,� entgegnete Nancy, �und sagt mir es lieber morgen. Ihr
    hattet mich auf 'nen Augenblick aufgest�rt, aber ich bin jetzt wieder
    so m�d' und d�mlich wie vorher.�

    Der Jude legte ihr noch mehrere andere Fragen in derselben Absicht
    vor, um zu erfahren, ob sie die ihm in einem unbewachten Augenblicke
    entschl�pften Andeutungen beachtet und verstanden h�tte; allein sie
    antwortete und hielt seine forschenden Blicke so unbefangen aus, da� er
    seinen ersten Gedanken, da� sie zuviel getrunken, vollkommen best�tigt
    zu sehen glaubte. Und Mi� Nancy war allerdings nicht frei von der unter
    Fagins Z�glingen gew�hnlichen Schw�che, der Neigung zum �berm��igen
    Genu� geistiger Getr�nke, in der sie in ihren zarteren Jahren eher
    best�rkt wurden, als da� man sie davon zur�ckgehalten h�tte. Ihr
    w�stes Aussehen und der das Gemach anf�llende starke Genevergeruch
    dienten zum bekr�ftigenden Beweise der Richtigkeit der Annahme des
    Juden; und als sie endlich zu weinen und gleich darauf wieder zu
    lachen anfing und wiederholt rief: �Heisa, wer wollte den Kopf h�ngen
    lassen!� so zweifelte er, der in Sachen dieser Art seinerzeit gro�e
    eigene Erfahrungen gemacht hatte, nicht mehr und freute sich h�chlich
    der Gewi�heit, da� ihre Trunkenheit in der Tat schon einen hohen Grad
    erreicht hatte.

    Er empfand infolge dieser Entdeckung eine gro�e Erleichterung und
    entfernte sich sehr zufrieden, seinen doppelten Zweck erreicht zu
    haben, dem M�dchen zu hinterbringen, was ihm von Toby mitgeteilt
    worden war, und sich mit eigenen Augen zu �berzeugen, da� Sikes nicht
    zur�ckgekehrt w�re. Es war eine Stunde vor Mitternacht und bitterlich
    kalt; er s�umte daher nicht, seine Wohnung baldm�glichst zu erreichen.
    Als er an der Ecke der Stra�e, in welcher sie lag, angelangt war und
    schon in der Tasche nach dem Hausschl�ssel suchte, trat pl�tzlich und
    unh�rbar ein Mann hinter ihn und fl�sterte seinen Namen dicht an
    seinem Ohre. Er wendete sich rasch um und sagte: �Ist das --�

    �Ja, ich bin's�, unterbrach ihn der Mann barsch. �Hab' hier seit zwei
    Stunden aufgepa�t. Wo zum Teufel seid Ihr gewesen?�

    �Besch�ftigt mit Euren Angelegenheiten, mein Lieber�, erwiderte der
    Jude, ihn unruhig anblickend und einen langsameren Schritt annehmend.
    �Den ganzen Abend besch�ftigt mit Euren Angelegenheiten.�

    �Ei, nat�rlich�, sagte der andere h�hnisch. �Was habt Ihr denn
    ausgerichtet?�

    �Nicht viel Gutes�, antwortete Fagin.

    �Ich will hoffen, nichts Schlimmes�, fiel der Vermummte, stillstehend
    und den Juden wild ansehend, ein.

    Fagin sch�ttelte den Kopf und stand im Begriff, ihm eine Antwort zu
    geben, als ihn der Vermummte unterbrach und sagte, er wolle lieber
    drinnen im Hause anh�ren, was er w�rde h�ren m�ssen, denn er w�re halb
    erfroren. Der Jude sah ihn mit einer Miene an, die offenbar genug
    verk�ndete, da� er des Besuches zu einer so sp�ten Stunde gar gern
    �berhoben w�re, und murmelte, da� er kein Feuer habe, und �hnliches;
    allein der unwillkommene Gast wiederholte seine Erkl�rung, mit ihm
    gehen zu wollen, mit gro�er Bestimmtheit, und Fagin schlo� die Haust�r
    auf und sagte ihm, er m�ge sie leise wieder verschlie�en, w�hrend er
    selbst Licht holen wolle.

    �'s ist hier so finster wie im Grabe�, bemerkte der Besucher, ein paar
    Schritte vorw�rts tappend. �Macht geschwind, ich kann solche Dunkelheit
    nicht leiden.�

    �Verschlie�t die T�r�, fl�sterte Fagin unten auf dem Hausflur, und
    w�hrend er sprach, wurde die T�re mit donnerndem Schalle zugeworfen.

    �Das hab' ich nicht getan�, sagte Fagins Peiniger, sich vorw�rts
    f�hlend. �Der Wind schlug sie zu, oder sie schlo� sich von selber.
    Macht geschwind, da� Ihr Licht bekommt, oder ich sto�e mir in diesem
    verw�nschten Loche den Kopf noch ein.�

    Fagin schlich in die K�che hinunter und kehrte bald darauf mit einem
    angez�ndeten Lichte und der Kunde zur�ck, da� Toby Crackit unten im
    Hinter- und die Knaben im Vorderzimmer schliefen. Er winkte seinem
    ungebetenen Gaste und f�hrte ihn die Treppe hinauf in ein Zimmer des
    oberen Stockwerks.

    �Wir k�nnen sagen hier die paar Worte, die wir haben zu sagen,� begann
    er, als sie eingetreten waren, �und ich will das Licht setzen drau�en
    an die Treppe, denn in den Fensterl�den sind L�cher, und wir lassen
    niemals sehen die Nachbarn, da� wir Licht haben.�

    Er stellte den Leuchter der T�r des Zimmers gegen�ber, in welchem sich
    nur ein gebrechlicher Sessel und hinter der T�r ein altes Sofa ohne
    �berzeug befand, auf das sich der m�de Fremde warf. Der Jude setzte
    sich vor ihn in den Sessel. Da die T�r halb offen stand, so war es im
    Zimmer nicht ganz finster, und das drau�en stehende Licht warf einen
    schwachen Schein auf die Wand gegen�ber.

    Sie fl�sterten einige Zeit so leise miteinander, da� ein Horcher
    von ihrer Unterredung nur etwa so viel h�tte verstehen k�nnen, um
    daraus zu entnehmen, da� sich Fagin gegen Beschuldigungen des Fremden
    verteidigte, und da� sich dieser in einer sehr gereizten Stimmung
    befand. Sie mochten etwa eine Viertelstunde gefl�stert haben, als Monks
    -- denn so hatte der Jude seinen Besucher mehrere Male genannt -- etwas
    lauter sagte: �Ich wiederhol's Euch, es war schlecht ausgedacht. Warum
    habt Ihr ihn nicht hier behalten bei den anderen und ohne weiteres 'nen
    j�mmerlichen Taschendieb aus ihm gemacht?�

    �H�r' einer an!� rief der Jude achselzuckend aus.

    �Wollt Ihr damit sagen, da� Ihr's nicht gekonnt h�ttet, wenn Ihr
    gewollt?� fragte Monks unwillig. �Habt Ihr's nicht bei hundert anderen
    Knaben verstanden? H�ttet Ihr h�chstens zehn bis zw�lf Monate Geduld
    gehabt, so w�r's Euch doch ein leichtes gewesen, zu machen, da� er
    verurteilt und vielleicht auf Lebenszeit deportiert wurde.�

    �Wem w�rde dabei gewesen sein gedient, mein Lieber?� fragte der Jude im
    dem�tigsten Tone.

    �Mir!�

    �Aber mir nicht�, fuhr Fagin fast noch unterw�rfiger fort. �Wenn zwei
    Leute sind beteiligt bei einem Gesch�ft, so ist's doch nur billig, da�
    ber�cksichtigt wird der Vorteil beider.�

    �Was weiter?�

    �Ich sah, da� es nicht leicht war, ihn zu erziehen zum Gesch�ft; er
    hatte nicht denselben Charakter wie andere Knaben.�

    �Hol' ihn der Satan, nein! denn er w�re sonst schon l�ngst ein
    Spitzbube gewesen.�

    �Ich hatte kein Mittel in H�nden, ihn zu machen schlimmer�, fuhr der
    Jude, angstvoll Monks Mienen beobachtend, fort; �er hatte in nichts die
    Hand drin; ich konnt' ihm mit gar nichts einjagen Furcht und Schrecken,
    und wir arbeiten immer vergeblich, wenn das nicht angeht. Was konnt'
    ich tun? Ihn ausschicken mit dem Baldowerer und Charley? Es geschah,
    und wir hatten genug an dem einen Male, mein Bester; ich mu�te zittern
    f�r uns alle.�

    �*Das* war meine Schuld nicht�, bemerkte der finstere Monks.

    �Freilich; nein, o nein, mein Lieber, und ich mache Euch auch keinen
    Vorwurf deshalb; denn w�r's nicht geschehen, so w�ren Eure Blicke
    vielleicht nicht gefallen auf den Knaben, und wir h�tten vielleicht
    niemals gemacht die Entdeckung, da� er es war, den Ihr suchtet. Nun
    gut; ich bracht' ihn wieder in meine Gewalt durch die Nancy, und jetzt
    f�ngt sie an und wirft sich auf zu seiner Freundin.�

    �Schn�rt ihr die Kehle zu!� sagte Monks ungeduldig.

    �Geht jetzt eben nicht an, mein Lieber,� versetzte Fagin l�chelnd; �und
    au�erdem machen wir in dergleichen keine Gesch�fte, sonst w�r mir's
    schon lieb, wenn es gesch�he �ber kurz oder lang. Monks, ich kenne
    diese Dirne, sobald anf�ngt der Knabe verh�rtet zu werden, wird sie
    sich nicht k�mmern um ihn mehr, als um 'nen Holzblock. Ihr wollt, da�
    er werden soll ein Dieb; ist er noch am Leben, so kann ich ihn jetzt
    dazu machen; und wenn -- wenn -- 's ist freilich nicht wahrscheinlich
    -- aber wenn sich das Schlimmste hat ereignet, und er ist tot --�

    �Wenn er's ist, so ist's meine Schuld nicht!� unterbrach ihn Monks mit
    best�rzter Miene und mit bebender Hand den Juden beim Arme fassend.
    �Merkt wohl, Fagin! ich habe keine Hand dabei im Spiel gehabt. Ich
    hab's Euch von Anfang an gesagt, alles -- nur nicht, da� er sterben
    sollte. Ich mag kein Blut vergie�en -- es kommt stets heraus und
    peinigt einen au�erdem! Ist er totgeschossen, so kann ich nichts daf�r;
    h�rt Ihr, Fagin? -- Was -- ist der Teufel in dieser verw�nschten
    Spelunke los? -- was war das?�

    �Was -- in aller Welt?� schrie der Jude, Monks mit beiden Armen
    umfassend, als derselbe pl�tzlich im h�chsten Schrecken emporsprang.
    �Was -- wo?�

    �Dort!� erwiderte der bebende Monks, nach der Wand gegen�ber
    hinzeigend. �Der Schatten -- ich sah den Schatten eines Frauenzimmers
    in 'nem Mantel und Hut, wie 'nen Hauch an dem T�felwerk dahingleiten.�

    Der Jude lie� ihn los, und beide st�rzten aus dem Zimmer hinaus.
    Das vom Zugwinde flackernde Licht, das an der Stelle stand, wo es
    Fagin hingestellt hatte, zeigte ihnen nur die leere Treppe und
    ihre erbleichten Gesichter. Sie horchten mit der gespanntesten
    Aufmerksamkeit, allein die tiefste Stille herrschte im ganzen Hause.

    �'s ist nichts gewesen als Eure Einbildung�, sagte der Jude, das Licht
    aufhebend und zu Monks sich wendend.

    �Ich will darauf schw�ren, da� ich's wirklich sah�, versetzte Monks,
    fortw�hrend heftig zitternd. �Es beugte sich vor, als ich's erblickte,
    und verschwand, als ich zu Euch davon zu sprechen anfing.�

    Der Jude warf ihm einen ver�chtlichen Blick zu, forderte ihn auf, ihm
    zu folgen, wenn es ihm beliebe, und ging voran die Treppe hinauf. Sie
    schauten in alle Gem�cher hinein, begaben sich wieder hinunter auf den
    Hausflur, in die Keller, durchsuchten jeden Winkel, allein vergebens.
    Es war im ganzen Hause �de und still wie der Tod.

    �Was meint Ihr nun, mein Guter?� sagte der Jude, als sie wieder auf dem
    Hausflur standen. �'s ist im Hause kein lebendiges Wesen au�er uns und
    Toby Crackit und den Knaben, und die sind wohl verwahrt. Schaut!�

    Er nahm zwei Schl�ssel aus der Tasche und f�gte hinzu, da� er, als
    er zuerst hinuntergegangen, Toby, Dawkins und Charley eingeschlossen
    habe, um jede St�rung des Gespr�chs unm�glich zu machen. Monks
    wurde wankend in seinem Glauben und erkl�rte endlich, da� ihm seine
    erhitzte Einbildungskraft einen Streich gespielt haben m�sse, wollte
    die Unterredung jedoch f�r diesmal nicht fortsetzen, erinnerte sich
    pl�tzlich, da� ein Uhr vor�ber sei, und das liebensw�rdige Freundespaar
    trennte sich.




    27. Kapitel.

        In dem die Unh�flichkeit eines fr�heren Kapitels bestm�glich wieder
        gutgemacht wird.


    Da es der geringen Person eines Schriftstellers schlecht anstehen
    w�rde, einen so wichtigen Mann wie einen Kirchspieldiener mit den
    Rocksch��en unter dem Arme am Feuer stehen zu lassen, bis es dem Autor
    eben beliebte, ihn zu erl�sen; und da ihm seine Stellung oder seine
    Galanterie noch weniger erlaubt, auf �hnliche Weise eine Dame zu
    vernachl�ssigen, auf welche besagter Kirchspieldiener ein wohlgeneigtes
    und z�rtliches Auge geworfen und in deren Ohren er s��e Worte
    gefl�stert, welche, aus dem Munde eines solchen Mannes kommend, in den
    Herzenssaiten jeglicher Jungfrau oder Matrone Anklang finden mu�ten: so
    eilt der gewissenhafte Erz�hler dieser Geschichte, der die geb�hrende
    Ehrfurcht vor denjenigen hegt, welche mit hoher und wichtiger
    Autorit�t bekleidet sind, ihnen jene Achtung zu zollen, welche ihre
    Stellung erfordert, und ihnen die ganze pflichtm��ige, r�cksichtsvolle
    Behandlung angedeihen zu lassen, zu welcher ihr hoher Rang und folglich
    ihre gro�en Tugenden sie auf das vollkommenste berechtigen. Es war
    seine Absicht, zu diesem Zwecke hier eine Abhandlung einzuf�gen, in
    welcher das g�ttliche Recht der Kirchspieldiener er�rtert und der Satz,
    da� ein Kirchspieldiener kein Unrecht tun k�nne, ins Licht gestellt
    werden sollte, -- eine Abhandlung, die f�r den verst�ndigen und
    wohlgesinnten Leser sowohl angenehm wie n�tzlich h�tte werden m�ssen;
    allein der Mangel an Zeit und Raum n�tigt ihn ungl�cklicherweise,
    sie f�r jetzt noch zur�ckzustellen. Sobald es indes an Zeit und Raum
    nicht mehr gebricht, wird er zeigen, da� ein Kirchspieldiener in der
    wahren und h�chsten Potenz -- das will sagen ein solcher, der beim
    Kirchspielarmenhause angestellt ist und in seiner amtlichen Eigenschaft
    die Kirchspielkirche besucht -- nach den Rechten und kraft seines Amtes
    alle Vortrefflichkeiten und mit einem Worte die besten Eigenschaften
    der menschlichen Natur besitzt, und da� blo�e Vereins- oder Kapellen-
    oder Gerichtsdiener oder Pedelle auf jene Vortrefflichkeiten auch nur
    die mindesten begr�ndeten Anspr�che keineswegs machen k�nnen.

    Mr. Bumble hatte wiederholt die Teel�ffel gez�hlt, die Zuckerzange
    gewogen, den Milchgie�er gepr�ft und s�mtliche Mobilien bis auf die
    Pferdehaarkissen der St�hle einer genauen Besichtigung unterworfen,
    ehe er daran dachte, da� es nachgerade wohl Zeit w�re, da� Mrs.
    Corney zur�ckkehrte. Sie lie� jedoch noch immer nichts von sich weder
    sehen noch h�ren, ein Gedanke pflegt einen anderen hervorzurufen,
    und so dachte Mr. Bumble weiter, da� er sich zum Zeitvertreibe
    nicht unschuldiger und gottseliger besch�ftigen k�nne, als wenn er
    seine Neugier durch einen fl�chtigen Blick in Mrs. Corneys Kommode
    befriedigte.

    Nachdem er daher an der T�r gehorcht hatte, ob auch niemand in der N�he
    w�re, fing er seine Untersuchung bei der untersten Schublade an, und
    die Kleider aus guten Stoffen, welche er fand, schienen ihm ausnehmend
    zu gefallen. In der obersten entdeckte er eine verschlossene B�chse,
    die er sch�ttelte, und das Geldgeklapper deuchte seinen Ohren gar
    liebliche Musik. Nachdem er sich eine Zeitlang daran erg�tzt hatte,
    stellte er sich wie zuvor an den Kamin und sagte mit feierlich-ernster
    Miene: �Ich tu's�, schien durch ein schlaues, wohlgef�lliges L�cheln
    hinzuf�gen zu wollen, was er doch f�r ein r�stiger, lustiger und
    pfiffiger alter Knabe sei, und betrachtete endlich mit vielem Vergn�gen
    und Interesse seine Waden im Profil.

    Er war noch in sotane, befriedigende Wadenschau vertieft, als Mrs.
    Corney hastig hereintrat, sich atemlos auf einen Stuhl am Kamin warf,
    mit der einen Hand die Augen bedeckte, die andere auf das Herz legte
    und nach Atem rang.

    �Mrs. Corney,� sagte Bumble, sich �ber sie beugend, �was ist Ihnen,
    Ma'am? Hat sich ein Ungl�ck ereignet? Ich bitte, antworten Sie mir; ich
    stehe hier auf -- auf --� Mr. Bumble konnte sich in seiner Best�rzung
    nicht auf das Wort �Kohlen� besinnen, er sagte daher: �wie auf
    Zuckerzangen�.

    �Oh, Mr. Bumble,� rief die Dame aus, �ich bin ganz wie zerschlagen!�

    �Zerschlagen -- wie?� z�rnte Bumble. �Wer hat sich unterfangen -- ah,
    ich wei� es schon,� f�gte er mit angeborener W�rde und Feierlichkeit
    hinzu, �abermals so ein St�ck von den spitzb�bischen, gottvergessenen
    Armen!�

    �'s ist schrecklich, nur daran zu denken!� sagte die Dame schaudernd.

    �So denken Sie nicht daran, Ma'am�, sagte Bumble.

    �Ich kann's nicht lassen�, entgegnete Frau Corney zimperlich.

    �So st�rken Sie sich durch einen Tropfen Wein�, riet der
    Kirchspieldiener in mitleidigem Tone.

    �Nicht um die Welt!� erwiderte Mrs. Corney. �Es w�re mir ganz
    unm�glich! Geistige Getr�nke -- nein -- nie -- Ach, ach! auf dem
    obersten Simse rechter Hand; ach, ach!�

    Die gute Frau hatte offenbar heftige Kr�mpfe und hatte schon die
    Besinnung verloren, als sie nach dem Eckschranke hinwies. Bumble flog
    auf denselben zu, fand eine gr�ne Flasche darin, nahm sie heraus,
    f�llte eine Tasse mit ihrem Inhalt und hielt sie der Dame an die Lippen.

    �Mir ist wohler!� sagte Mrs. Corney, nachdem sie die Arznei halb
    ausgetrunken hatte.

    Bumble hob zum Zeichen seiner dankbaren Gef�hle die Augen zur Decke
    empor, senkte sie nieder auf den Rand der Tasse und hielt dieselbe
    unter seine Nase.

    �Pfefferminzwasser�, sagte Mrs. Corney mit matter Stimme, aber dem
    Kirchspieldiener zul�chelnd. �Kosten Sie doch einmal -- es ist noch ein
    wenig sonst was drin.�

    Bumble kostete den heilkr�ftigen Trank, kostete noch einmal mit weiser,
    pr�fender Miene, und stellte die Tasse leer auf den Tisch.

    �Es bekommt vortrefflich�, bemerkte die Patientin.

    Bumble erkl�rte, derselben Meinung zu sein, setzte sich neben Frau
    Corney und fragte z�rtlich: �Was ist Ihnen aber begegnet, Ma'am?�

    �O nichts�, erwiderte sie; �ich bin eine recht t�richte, erregbare,
    schwache Frau.�

    �Schwach, Ma'am�, sagte Bumble, ein wenig n�her r�ckend. �Sind Sie
    wirklich schwach, Mrs. Corney?�

    �Wir sind alle schwache Gesch�pfe�, versetzte Mrs. Corney, einen
    allgemeinen Satz aufstellend.

    �Sehr wahr�, stimmte Bumble ein.

    Ein paar Minuten lang schwiegen beide, und nach Ablauf derselben hatte
    Mr. Bumble den allgemeinen Satz praktisch dadurch erl�utert, da� er
    seinen linken Arm von Mrs. Corneys Stuhllehne entfernt und um ihr
    Sch�rzenband gelegt, wo er nunmehr mit sanftem Drucke ruhte.

    �Wir sind allesamt schwache Gesch�pfe�, wiederholte er.

    Mrs. Corney seufzte.

    �Seufzen Sie doch nicht, Ma'am!�

    �Ach! Wenn ich's nur lassen k�nnte!� Sie seufzte abermals.

    �Dies Zimmerchen ist sehr nett und behaglich, Ma'am. Es w�rde mit noch
    so einem eine artige Wohnung ausmachen.�

    �Es w�rde zu viel sein f�r eine einzelne Person�, murmelte die Dame.

    �Aber nicht f�r zwei, Ma'am�, fiel Bumble schmachtend ein. �Was sagen
    Sie, Mrs. Corney?�

    Mrs. Corney senkte den Kopf bei diesen Worten Mr. Bumbles, und Mr.
    Bumble senkte den seinigen gleichfalls, um ihr in das Gesicht schauen
    zu k�nnen. Mrs. Corney blickte mit gro�er Z�chtigkeit seitw�rts, machte
    ihre Hand los, um nach ihrem Taschentuche zu greifen, und lie� sie
    unwillk�rlich in die Hand Mr. Bumbles sinken.

    �Gibt Ihnen die Direktion nicht freie Feuerung, Mrs. Corney?� fragte
    der Kirchspieldiener, ihr z�rtlich die Hand dr�ckend.

    �Und freies Licht�, erwiderte Mrs. Corney, den Druck leise erwidernd.

    �Feuerung, Licht und Wohnung frei�, fuhr Bumble fort. �Oh, Mrs. Corney,
    welch ein Engel Sie sind!�

    Die Dame war gegen einen solchen Gef�hlsergu� nicht unempfindlich
    genug, um noch l�nger widerstehen zu k�nnen, sondern sank in die Arme
    Mr. Bumbles, welcher Gentleman ihr im Sturme seiner Gef�hle einen
    leidenschaftlichen Ku� auf die keusche Nase dr�ckte.

    �Oh, Sie Ausbund aller Kirchspielvollkommenheiten!� rief Mr. Bumble
    ganz verz�ckt aus. �Sie wissen doch, meine Himmlische, da� Mr. Slout
    heut' abend viel kr�nker geworden ist?�

    �Ach ja�, sagte Mrs. Corney versch�mt.

    �Der Doktor sagt, da� er keine acht Tage mehr leben k�nnte�, fuhr
    Bumble fort. �Sein Tod hat die Vakanz des Haushofmeisterpostens zur
    Folge. Oh, Mrs. Corney, welche Aussichten er�ffnen sich da! -- welche
    Aussichten auf die allerseligste Herzens- und Haushaltsverschmelzung!�

    Mrs. Corney schluchzte.

    �O meine bezaubernde Mrs. Corney!� sprach Bumble weiter, �das kleine
    W�rtchen -- nur das kleine, s��e W�rtchen!�

    �Ja -- a -- a!� hauchte Mrs. Corney.

    �Und noch eins -- nur das eine noch -- wann soll es sein?�

    Sie versuchte zweimal zu reden, doch vergebens. Endlich fa�te sie
    sich ein Herz, schlang die Arme um Bumbles Nacken und sagte, sobald
    es ihm nur irgend gefiele, und er w�re ein gar zu lieber und ganz
    unwiderstehlicher Mann.

    Nachdem die Angelegenheit auf diese freundschaftliche und befriedigende
    Weise geordnet war, wurde der Vertrag durch eine zweite Tasse
    Pfefferminzwasser feierlich besiegelt, was bei der Erregtheit und
    Beklemmung der Dame um so notwendiger war; und w�hrend die Tasse
    geleert wurde, erz�hlte Mrs. Corney ihrem Zuk�nftigen von dem Tode der
    alten Frau.

    �Sch�n�, bemerkte Bumble, sein Pfefferminzwasser schl�rfend. �Ich will
    auf meinem R�ckwege nach Hause bei Sowerberry vorsprechen und die
    erforderlichen Anordnungen treffen. Was war es denn aber, wor�ber Sie
    so ganz au�er sich zu sein schienen, meine Liebe?�

    �Oh, es war nichts Besonderes, Bester�, erwiderte die Dame ausweichend.

    �Ei, es mu� doch etwas Besonderes gewesen sein. Warum wollen Sie es
    Ihrem Bumble nicht sagen?�

    �Ein anderes Mal -- wenn wir erst verheiratet sind, mein Teuerster.�

    �Wenn wir erst verheiratet sind! Es wird sich doch kein Armer eine
    Unversch�mtheit gegen Sie herausgenommen haben?�

    �O nein, nein, durchaus nicht!� fiel die Dame hastig ein.

    �Wenn ich das auch annehmen m��te,� fuhr der Kirchspieldiener fort,
    �denken m��te, da� es ein Armer gewagt h�tte, seine gemeinen Augen zu
    dem liebensw�rdigen Antlitze zu erheben --�

    �Das h�tte keiner gewagt -- nimmermehr --�

    �Ich wollt's ihnen auch wohl raten!� z�rnte Bumble, die Faust
    sch�ttelnd. �Ich will den Menschen sehen, arm oder nicht arm, der
    sich's unterfinge, und kann ihm nur so viel versichern, da� er's nicht
    zum zweitenmal tun w�rde.�

    Die Worte h�tten vielleicht wie eine nicht eben gro�e Schmeichelei
    gegen die Reize der Dame geklungen, wenn sie nicht durch heftiges
    Geb�rdenspiel versch�nt gewesen w�ren; da jedoch Bumble seine Drohung
    mit vielen kriegerischen Gestikulationen begleitete, so erblickte
    Mrs. Corney darin sehr ger�hrt nur einen Beweis seiner aufopfernden
    Ergebenheit und versicherte ihm bewundernd und mit gro�er W�rme, da� er
    wahrhaftig ein T�ubchen w�re.

    Mr. Bumble kn�pfte den Rock bis unter das Kinn zu, setzte seinen
    dreieckigen Hut auf, umarmte seine Taube z�rtlich und lange und ging,
    um abermals dem Sturme und der K�lte Trotz zu bieten, nachdem er zuvor
    blo� noch f�nf Minuten im Zimmer der m�nnlichen Armen verweilt und
    gegen dieselben ein wenig getobt hatte, um zu erproben, ob er der
    Stelle des Haushofmeisters auch mit der geb�hrenden Autorit�t w�rde
    vorstehen k�nnen. Nachdem er sich von seiner Bef�higung �berzeugt,
    verlie� er das Haus mit einem leichten, fr�hlichen Herzen und
    gl�nzenden Vorausahnungen seiner bevorstehenden Bef�rderung.

    Mr. und Mrs. Sowerberry befanden sich in einer Abendgesellschaft, und
    da Noah Claypole zu keiner Zeit geneigt war, sich einem gr��eren Ma�e
    physischer Anstrengung zu unterziehen, als durch eine gem�chliche
    Bet�tigung der Funktionen des Essens und Trinkens erfordert wird, so
    war der Laden noch nicht verschlossen, obgleich die Stunde l�ngst
    vor�ber war, zu welcher es h�tte geschehen sollen. Bumble klopfte
    mehreremal mit seinem Rohre auf den Ladentisch; allein da niemand
    erschien, und da er durch das Glasfenster des kleinen Zimmers hinter
    dem Laden Licht schimmern sah, so trat er n�her, um nachzusehen, was in
    dem Zimmerchen vorginge, und war nicht wenig erstaunt, zu sehen, was er
    sah.

    Der Tisch war gedeckt, und auf ihm standen Brot und Butter, Teller
    und Gl�ser, ein Krug mit Porter und eine Weinflasche. Noah Claypole
    ruhte in nachl�ssigster Stellung in einem Sessel und hatte ein
    m�chtiges Butterbrot in der Hand. Dicht neben ihm stand Charlotte
    und �ffnete Austern, welche Noah sich herablie�, mit gro�em Behagen
    zu verschlingen. Eine mehr als gew�hnliche R�te in der Gegend der
    Nase des jungen Herrn und ein gewisses Blinzeln seines rechten
    Auges verk�ndigten, da� er ein wenig angetrunken war, und die
    besagten Symptome erhielten noch eine Verdeutlichung durch seine
    augenscheinliche Begier nach den Austern, die er offenbar haupts�chlich
    wegen ihrer k�hlenden Eigenschaften bei innerlicher Glut geno�.

    �Da ist 'ne pr�chtige, fette, Noah!� sagte Charlotte. �Die mu�t du
    probieren.�

    �Wie wundervoll doch Austern schmecken!� bemerkte Noah; �und wie schade
    ist's, da� man sich immer unbehaglich f�hlt, wenn man sie in einiger
    Menge genossen hat.�

    �'s ist wirklich grausam und unrecht�, sagte Charlotte. �Hier ist
    wieder 'ne ganz herrliche.�

    �Tut mir leid, ich kann nicht mehr. Komm her, Charlotte, da� ich dich
    k�sse�, sagte Noah.

    �Wie -- was?� schrie Bumble, hineinst�rzend. �Sag das noch einmal,
    Bursch!�

    Charlotte stie� einen Schrei aus und verbarg ihr Gesicht hinter
    der Sch�rze, w�hrend Noah, ohne seine Lage zu ver�ndern, den
    Kirchspieldiener mit dem Starrblicke der Trunkenheit angaffte.

    �Sag' das noch einmal, du sch�ndlicher, schamloser Schlingel!� fuhr
    Bumble fort. �Wie kannst du es wagen, von K�ssen zu sprechen? Und Sie,
    freches Weibsbild, wie unterstehen Sie sich, ihn dazu aufzumuntern?
    K�ssen! Pfui!� rief er in starker und gerechter Entr�stung aus.

    �Ich wollt' es gar nicht!� sagte Noah best�rzt und flehend. �Sie k��t
    mich immer, ich mag es haben wollen oder nicht.�

    �O Noah!� rief Charlotte mit einem Blicke des Vorwurfs.

    �Ja, es ist wahr,� sprudelte Noah, �du tust's immer. Mr. Bumble, sie
    l��t's und l��t's nicht und klopft mich immer unter das Kinn und
    flattiert mir auf alle ersinnliche Weise.�

    �Schweig!� donnerte Bumble. �Sie packen sich sogleich hinaus, und du,
    Musj� Noah, verschlie�t den Laden und sprichst, bis dein Herr nach
    Hause kommt, kein Wort mehr, auf deine eigene Gefahr; und wenn er nach
    Hause kommt, so sag ihm, ich lie�e ihm sagen, er m�chte morgen fr�h
    nach dem Fr�hst�ck 'nen Sarg f�r 'ne alte Frau schicken. H�rst du? --
    K�ssen! Die S�ndhaftigkeit und Gottlosigkeit der geringeren Klasse
    in diesem Kirchspielbezirke hat eine schreckliche H�he erreicht, und
    zieht das Parlament ihre Verdorbenheit nicht in Betracht, so ist das
    Land zugrunde gerichtet und die Sittlichkeit des Volkes f�r immer zum
    Henker!�

    Mit diesen Worten schritt er majest�tisch und d�ster hinaus; und da
    wir ihn nun so weit auf seinem Heimwege begleitet und alle n�tigen
    Anordnungen zum Begr�bnisse der alten Frau getroffen haben, wollen wir
    uns nach Oliver Twist umsehen und unsere Wi�begier befriedigen, ob er
    noch in dem Graben liegt, in welchem Bill Sikes und Toby Crackit ihn
    haben liegen lassen.




    28. Kapitel.

        Was Oliver nach dem mi�lungenen Einbruche begegnete.


    �Da� euch die W�lfe zerrissen!� murmelte Sikes z�hneknirschend.
    �Wollte, da� ich einem von euch nahe genug w�re, er sollte mir erst
    Ursache zum Heulen bekommen!�

    Indem Sikes mit dem w�tendsten Ingrimme, dessen er f�hig war, diese
    Worte vor sich hin sprach, legte er den verwundeten Knaben �ber sein
    niedergebeugtes Knie und sah sich nach seinen Verfolgern um. Er
    vermochte in dem Nebel und der Finsternis nichts zu unterscheiden,
    allein desto heller und lauter t�nte das Rufen und Schreien der
    Nachsetzenden, das Gebell der Hunde rings umher und der Schall der
    L�rmglocke durch die Nacht.

    �Steh, feiger Schuft!� schrie Sikes Toby Crackit nach, der den
    eilfertigsten Gebrauch von seinen langen Beinen zu machen angefangen
    hatte, und schon eine Strecke voraus war. �Steh' augenblicklich!�

    Toby gehorchte, da er noch nicht vollkommen gewi� war, au�er Schu�weite
    zu sein, und deutlich erkannte, da� Sikes nicht in der Stimmung w�re,
    mit sich scherzen zu lassen.

    �Hilf mir den Knaben forttragen�, tobte der W�tende. �Komm zur�ck --
    hierher!�

    Toby kehrte langsam einige Schritte zur�ck, wagte indes leise und
    atemlos einige bescheidene Gegenvorstellungen.

    �Geschwinder!� schrie Sikes, legte den Knaben in einen trockenen Graben
    und zog eine Pistole hervor. �Hab' mich ja nicht zum Narren!�

    Gerade in diesem Augenblick verdoppelte sich der L�rm, und Sikes konnte
    erkennen, da� die Verfolger bereits �ber die Umz�unung des Feldes
    kletterten, auf welchem er sich mit Toby und Oliver befand, und da�
    ihnen ein paar Hunde mehrere Schritte voraus waren.

    �'s ist nichts mehr zu machen, Bill�, sagte Toby; �la�t den
    Schreiling[AM] und nehmt die Bein' untern Arm!�

      [AM] Knabe.

    Er mochte sich lieber der M�glichkeit aussetzen, von dem Freunde
    niedergeschossen zu werden, als unfehlbar dem Feinde in die H�nde zu
    fallen, und rannte daher, so schnell ihn seine F��e tragen wollten,
    davon. Sikes bi� die Z�hne zusammen, warf einen Tuchkragen �ber den
    Knaben, lief an der n�chsten Hecke hin, um die Verfolger zu t�uschen,
    stand vor einer zweiten still, die mit jener in einem rechten Winkel
    zusammenstie�, schleuderte seine Pistole hoch in die Luft, wagte einen
    verzweifelten Sprung und rannte in einer anderen Richtung als Toby fort.

    Seine Eile war unn�tig, denn w�hrend er �ber Stock und Block
    davoneilte, rief schon einer der drei Nachsetzenden die Hunde zur�ck,
    die gleich ihren Herren kein gro�es Behagen an der Verfolgung zu finden
    schienen und daher augenblicklich gehorchten. Das Kleeblatt war nur
    wenige Schritte weit auf das Feld vorgedrungen und stand still, um zu
    beraten.

    �Mein Rat, oder wenigstens mein Befehl ist der,� sagte der dickste der
    drei M�nner, �da� wir auf der Stelle umkehren und wieder nach Hause
    gehen.�

    �Mir ist alles recht, was Mr. Giles recht ist�, fiel ein kleinerer Mann
    ein, der indes auch keineswegs schm�chtig genannt werden konnte und
    sehr bla� und sehr h�flich war, wie es die Leute h�ufig sind, wenn die
    Furcht sie beherrscht.

    �Meine Herren,� nahm der dritte das Wort, der die Hunde zur�ckgerufen
    hatte, �ich m�chte nicht gern ungezogen erscheinen. Mr. Giles mu� es am
    besten wissen.�

    �Ja, ja,� fiel der kleinere ein, �was Mr. Giles sagt, dem d�rfen wir
    nicht widersprechen; nimmermehr, ich kenne meine Stellung Gott sei Dank
    zu gut, um mir's herauszunehmen.�

    Der kleine Mann schien seine Stellung in der Tat nicht blo� genau
    zu kennen, sondern auch sehr unangenehm zu empfinden, denn er stand
    z�hneklappernd neben den beiden andern.

    �Sie f�rchten sich, Brittles�, sagte Mr. Giles.

    �Nicht im mindesten�, sagte Brittles.

    �Sie f�rchten sich allerdings!�

    �Sie irren, Mr. Giles.�

    �Sie l�gen, Brittles.�

    Das Zwiegespr�ch war eine Folge davon, da� Mr. Giles Verdru� empfand,
    und sein Verdru� war aus seinem Unwillen dar�ber entsprungen, da� die
    Verantwortlichkeit wegen der R�ckkehr nach Hause in der Form eines
    Kompliments auf ihn zur�ckgew�lzt worden war. Der dritte Mann beendete
    den Streit sehr philosophisch. �Lassen Sie mich Ihnen sagen, wie es
    ist�, fiel er ein; �wir f�rchten uns alle.�

    �Sie reden nach Ihrer eigenen Erfahrung�, versetzte Mr. Giles, der der
    bl�sseste von den dreien war.

    �Allerdings�, sagte der Angeredete. �'s ist unter solchen Umst�nden
    ganz nat�rlich und schicklich, da� man sich f�rchtet.�

    �Nun, ich f�rchte mich auch�, sagte Brittles; �aber warum ist es
    notwendig, es einem so geradezu in das Gesicht zu sagen?�

    Diese offenen Gest�ndnisse bes�nftigten Mr. Giles, der sofort
    einr�umte, auch seinerseits einige Furcht zu empfinden, worauf alle
    drei in der vollkommensten Einm�tigkeit zur�ckzueilen anfingen. Nicht
    lange nachher trug jedoch Mr. Giles, der den k�rzesten Atem hatte und
    eine gro�e Heugabel trug, auf ein kurzes Verweilen an, um sich wegen
    seiner Ausf�lle zu entschuldigen.

    �Man glaubt es aber gar nicht,� schlo� er, �wozu man f�hig ist, wenn
    einem das Blut warm geworden. Wahrhaftig, ich w�rde einen Mord begangen
    haben -- ich wei� es -- h�tten wir einen der B�sewichter gefangen.�

    Die anderen beiden hatten �hnliche �berzeugungen und konnten nur nicht
    begreifen, wie es zugegangen, da� in ihrer Stimmung eine so pl�tzliche
    �nderung eingetreten war.

    �Ich wei� es�, sagte Giles; �es kam von der Umz�unung her. Ja, die
    Umz�unung des Feldes, auf welchem wir die Halunken fast ertappt h�tten,
    unterbrach alle Mordgedanken und hemmte die innere Wut. Ich f�hlte sie
    bei mir im Hin�bersteigen vergehen.�

    Durch ein merkw�rdiges Zusammentreffen hatten die beiden andern
    dasselbe Gef�hl genau in demselben Augenblick empfunden, so da� Mr.
    Giles ganz offenbar recht gehabt hatte, als er sagte: es kam von der
    Umz�unung her -- namentlich, da hinsichtlich des Zeitpunktes, in dem
    die Ver�nderung Platz gegriffen hatte, gar kein Zweifel obwalten
    konnte, da sich alle drei entsannen, da� sie in dem Augenblicke, da sie
    eintrat, die Einbrecher zu Gesicht bekommen h�tten.

    Dieses Gespr�ch f�hrten die beiden M�nner, welche die Einbrecher
    �berrascht hatten, und ein wandernder Kesselflicker, der in einem
    Nebengeb�ude geschlafen und sich nebst seinen beiden Hunden hatte
    entschlie�en m�ssen, an dem gefahrvollen Abenteuer der Diebesverfolgung
    teilzunehmen. Mr. Giles diente der alten Dame, welche das Haus
    bewohnte, in der doppelten Eigenschaft als Keller- und Haushofmeister,
    und Brittles war Bedienter, G�rtner, Ausl�ufer usw. Die alte Dame
    hatte ihn in ihren Dienst genommen, als er noch ein kleiner,
    vielversprechender Knabe gewesen war, und er wurde noch immer als ein
    solcher behandelt, obgleich er in den Drei�igern stand.

    Die drei k�hnen M�nner setzten unter ermutigenden und die Zeit
    k�rzenden angenehmen Gespr�chen in geschlossener Phalanx ihren R�ckzug
    fort und bewiesen, obwohl ihnen noch mancher ungew�hnlich starke
    Windsto� Schrecken einjagte, die Geistesgegenwart, ihre Laterne
    abzuholen, die sie hinter einem Baume hatten stehen lassen, damit sie
    den Dieben nicht anzeigte, wohin sie schie�en m��ten, falls sie etwa
    Feuer zu geben geneigt w�ren.

    Sie waren l�ngst zu Hause angelangt, die Luft wurde immer k�lter, je
    n�her der Morgen kam, der Nebel bewegte sich am Boden entlang wie
    eine dichte Rauchwolke, das Gras war feucht, die Fu�wege und niedrig
    gelegenen Stellen waren kotig und schlammig, ein na�kalter Wind
    verk�ndete sein Nahen durch ein hohles Brausen -- und Oliver lag noch
    immer bewu�tlos in dem Graben, wo ihn Sikes niedergelegt hatte. Im
    Osten zeigte sich das erste matte Morgengrauen -- eher dem Tode der
    Nacht als der Geburt des Tages zu vergleichen. Die Gegenst�nde, die in
    der Finsternis unheimlich ausgesehen hatten, nahmen immer bestimmtere
    Umrisse an und erhielten allm�hlich ihre gew�hnliche Gestalt. Der Regen
    rauschte in Str�men nieder und schlug klatschend auf die entlaubten
    B�sche -- aber Oliver empfand kein Ungemach davon; er lag fortw�hrend
    hilflos und ohnm�chtig auf seinem harten, feuchten Bette von Erde.

    Endlich weckte ihn ein empfindlicher Schmerz; er schrie laut auf und
    erwachte. Sein linker, in der Eile mit einem Tuche verbundener Arm
    hing schwer und gel�hmt an seiner Seite, und das Tuch war mit Blut
    getr�nkt. Er war so schwach, da� er sich kaum in eine sitzende Stellung
    emporzurichten vermochte; er blickte matt nach Hilfe umher und �chzte
    vor Schmerz. Bebend an allen Gliedern vor K�lte und Ersch�pfung, suchte
    er sich aufzurichten, fiel aber ohnm�chtig der L�nge nach wieder nieder.

    Eine ihn �berfallende Ohnmachtsempfindung, die ihm die Warnung
    zuzuraunen schien, er werde unfehlbar sterben m�ssen, wenn er noch
    l�nger dal�ge, brachte ihn zum Bewu�tsein zur�ck. Er stand m�hsam auf,
    ihm schwindelte jedoch, und er wankte gleich einem Betrunkenen von
    einer Seite zur anderen. Er hielt sich nichtsdestoweniger aufrecht und
    taumelte mit gesenktem Kopfe vorw�rts, ohne zu wissen wohin.

    In seinem Innern dr�ngten sich �ngstigende, verwirrte Gedanken und
    Bilder. Es war ihm, als wenn er noch zwischen Sikes und Crackit ginge,
    die ergrimmt miteinander zankten; ihre Worte t�nten noch in seinen
    Ohren, und als ihn ein Fehltritt zum Bewu�tsein zur�ckrief, machte er
    die Entdeckung, da� er zu den beiden schrecklichen M�nnern redete,
    als wenn er sich noch in ihrer Gewalt bef�nde. Dann kam es ihm wieder
    vor, als wenn er mit Sikes allein w�re, der ihm seine Rolle bei dem
    beabsichtigten Einbruche einzupr�gen suchte. Unbestimmte, d�stere
    Gestalten schwebten an ihm hin und wieder, er schreckte zusammen bei
    dem vermeintlichen Knalle eines Feuergewehrs, er h�rte lautes Rufen
    und Schreien, vor seinen Augen flimmerten und verschwanden Lichter,
    es summte ihm vor den Ohren, alles vor Verwirrung und L�rmen, er
    f�hlte sich durch eine unsichere Hand fortgetragen; und w�hrend er
    so halb wachend tr�umte, peinigte und �ngstigte ihn fortw�hrend ein
    undeutliches Schmerzgef�hl, ein Halbbewu�tsein seiner uns�glich
    jammervollen Lage.

    So wankte er weiter und weiter, fast mechanisch durch Gitter oder
    L�cken in den Hecken kriechend, bis er einen Weg erreicht hatte; und
    jetzt fing es an so stark zu regnen, da� er wirklich erwachte. Er
    blickte umher und sah in geringer Entfernung ein Haus, bis zu welchem
    er sich fortschleppen zu k�nnen meinte. Die Bewohner desselben hatten
    vielleicht Mitleid mit ihm, und wo nicht, so war es doch besser, wie er
    dachte, in der N�he menschlicher Wesen zu sterben als allein auf den
    �den Feldern. Er sammelte seine letzten Kr�fte, und eilte so rasch er
    konnte, dem Hause zu.

    Als er n�her kam, war es ihm, als wenn er es schon gesehen h�tte,
    wenn er sich auch seiner einzelnen Teile nicht erinnern konnte. Doch
    ach, die Gartenmauer! Und dort an jener Stelle hatte er sich in der
    vergangenen Nacht auf die Knie niedergeworfen und die beiden M�nner
    um Erbarmen angefleht. Es war dasselbe Haus, das sie zu berauben
    versucht hatten. Auf ein paar Augenblicke �berkam ihn ein Gef�hl so
    entsetzlichen Schreckens, da� er den Schmerz seiner Wunde verga� und
    nur an Flucht dachte. Flucht! Er war kaum imstande, sich auf den F��en
    zu halten, und h�tte er die Kr�fte dazu gehabt, wohin h�tte er fliehen
    sollen? Die Gartent�r stand offen, er wankte �ber den Grasplatz, stieg
    mit M�he die Stufen des Portals vor der Haust�r hinan und klopfte
    leise; die Kr�fte schwanden ihm, und er sank ohnm�chtig nieder.

    Gerade zu derselben Stunde st�rkten sich Mr. Giles, Brittles und der
    Kesselflicker nach den Strapazen und Schrecken der Nacht in der K�che
    durch ein Sch�lchen Tee, und was es sonst Gutes gab. Nicht, da� es
    Mr. Giles' Gewohnheit gewesen w�re, zu gro�e Vertraulichkeit mit der
    niederen Dienerschaft zu pflegen, gegen welche er sich vielmehr der
    Regel nach nur mit einer leutseligen Herablassung benahm, die stets an
    seine h�here Stellung in der Gesellschaft erinnerte. Allein Todesf�lle,
    Feuersbr�nste und Einbr�che machen alle Menschen gleich, und Mr.
    Giles sa� mit ausgestreckten F��en am Herde, hatte den linken Arm auf
    den Tisch gest�tzt, illustrierte mit dem rechten einen genauen und
    gl�henden Bericht �ber den n�chtlichen Raubanfall, und sein Publikum --
    zumal die K�chin und das Hausm�dchen -- h�rte ihm in atemloser Spannung
    zu.

    �Es mochte halb zwei Uhr sein,� sagte Mr. Giles, �indes kann ich nicht
    darauf schw�ren, ob's nicht dreiviertel war, als ich aufwachte, mich
    im Bett herumdrehte, ungef�hr so� (er drehte sich bei diesen Worten
    auf seinem Stuhle herum und zog den Zipfel des Tischtuches �ber die
    Schultern, um bildlich desto lebhafter die Vorstellung einer Bettdecke
    hervorzurufen), �und ein Ger�usch zu h�ren glaubte.�

    Hier erbla�te die K�chin und forderte das Hausm�dchen auf, die
    K�chent�r zu verschlie�en; das Hausm�dchen hie� es Brittles, der es den
    Kesselflicker hie�, der sich stellte, als ob er nicht h�rte.

    �Ein Ger�usch zu h�ren glaubte�, wiederholte Mr. Giles. �Ich dachte
    zuerst bei mir selbst: 's ist eine T�uschung! und legte mich schon
    wieder zum Einschlafen zurecht, als ich das Ger�usch abermals und
    vollkommen deutlich vernahm.�

    �Was war es denn f�r eine Art von Ger�usch?� fragte die K�chin.

    �Ein krachendes�, erwiderte Mr. Giles.

    �Ich denke, es war mehr, wie wenn eine eiserne Stange auf einem
    Reibeisen gerieben wird�, fiel Brittles ein.

    �So war es, als *Sie* es h�rten�, sagte Giles; �zu der Zeit aber, als
    ich es vernahm, hatte es einen krachenden Ton. Ich warf die Bettdecke
    zur�ck� (er wiederholte die Bewegung mit dem Tischtuche), �richtete
    mich zum Sitzen empor und horchte.�

    Die K�chin und das Hausm�dchen riefen zugleich aus: �Da� sich Gott
    erbarm'!� und r�ckten zusammen.

    �Ich h�rte es so deutlich, als wenn es dicht vor meinem Bette w�re�,
    fuhr Giles fort, �und dachte: Da wird eine T�r oder ein Fenster
    aufgebrochen. Was ist zu tun? Ich will den guten Jungen Brittles
    wecken und ihn retten, damit er nicht in seinem Bette ermordet wird;
    tu' ich's nicht, so kann ihm die Kehle vom rechten bis zum linken Ohre
    abgeschnitten werden, ohne da� er es merkt.�

    Hier richteten sich aller Blicke auf Brittles, der die seinigen auf den
    Redner heftete und ihn mit weit ge�ffnetem Munde anstarrte, w�hrend
    seine Mienen den grenzenlosesten Schrecken ausdr�ckten.

    �Ich stie� die Bettdecke von mir,� sprach Giles, das Tischtuch von sich
    schleudernd und die K�chin und das Hausm�dchen scharf fixierend, �stieg
    leise aus dem Bette, zog --�

    �Es sind Damen anwesend, Mr. Giles!� murmelte der Kesselflicker.

    �-- Meine Pantoffel an, Sir,� fuhr Giles, sich zu ihm wendend und
    den st�rksten Nachdruck auf seine Pantoffel legend, fort, �nahm
    die geladene Pistole zur Hand, die immer mit dem Silberzeugkorbe
    hinaufgebracht wird, ging auf den Zehen in Brittles Kammer und sagte,
    sobald ich ihn aus dem Schlafe ger�ttelt hatte: >Brittles, erschrecken
    Sie nicht!<�

    �Ja, das sagten Sie, Mr. Giles�, fiel Brittles mit bebender Stimme ein.

    �>Brittles, ich glaube, wir sind verloren<, sagt' ich,� fuhr Giles
    fort, �>aber seien Sie nur ohne Furcht.<�

    �Zeigte er denn auch keine Furcht?� fragte die K�chin.

    �Nein�, antwortete Giles; �er war so unverzagt -- fast so unverzagt wie
    ich selber.�

    �Ich w�re auf der Stelle gestorben, wenn ich's gewesen w�re�, bemerkte
    das Hausm�dchen.

    �Sie sind ein junges M�dchen�, fiel Brittles ein, der ziemlich herzhaft
    zu werden anfing.

    �Brittles hat recht�, sagte Giles mit beil�ufigem Kopfnicken; �von
    einem M�dchen war nichts anderes zu erwarten. Wir aber, als M�nner,
    nahmen eine Blendlaterne aus Brittles' Kammer und f�hlten uns in der
    Pechrabenfinsternis hinunter.� (Er war aufgestanden, hatte die Augen
    geschlossen, tappte ein paar Schritte vorw�rts und durchs�gte, um seine
    Schilderung mit angemessener Aktion zu begleiten, mit den Armen die
    Luft, bis er mit der K�chin in eine unangenehme Ber�hrung kam und die
    K�chin und das Hausm�dchen zu schreien anfingen, worauf er nach seinem
    Stuhle zur�ckeilte.) �Was hat das zu bedeuten?� unterbrach er sich
    pl�tzlich; �es wird geklopft -- �ffne jemand die Haust�r!�

    Niemand regte sich.

    �Das ist doch seltsam, da� zu einer so fr�hen Morgenstunde geklopft
    wird�, fuhr Giles, umherschauend und bleichen Antlitzes nur bleiche
    Gesichter gewahrend, fort; �allein die T�r mu� ge�ffnet werden. He!
    Holla! h�rt denn niemand?�

    Mr. Giles richtete bei diesen Worten die Blicke auf Brittles; allein
    der von Natur bl�de, bescheidene J�ngling hielt sich mutma�lich
    in der Tat f�r niemand und meinte sicher, da� die Frage unm�glich
    an ihn gerichtet sein k�nne. Jedenfalls gab er keine Antwort. Mr.
    Giles sendete dem Kesselflicker auffordernde Blicke zu, allein der
    Kesselflicker war urpl�tzlich eingeschlafen. Die Frauenzimmer kamen
    nicht in Betracht.

    �Wenn Brittles die T�r etwa lieber in Gegenwart von Zeugen �ffnen
    will,� sagte Mr. Giles nach einem kurzen Stillschweigen, �so bin ich
    bereit, einen solchen abzugeben.�

    �Ich auch�, sagte der Kesselflicker, ebenso pl�tzlich wieder
    aufwachend, wie er eingeschlafen war.

    Brittles kapitulierte auf diese Bedingungen, und als man beim �ffnen
    der Fensterl�den die Entdeckung machte, da� es heller Tag war, und
    dadurch gar sehr an Mut gewann, so zog die kleine, tapfere Schar aus,
    die Hunde voran und die Frauenzimmer in der Nachhut. Gem�� dem Rate Mr.
    Giles' sprachen alle sehr laut, um dem Feinde sogleich kundzutun, wie
    zahlreich sie waren, und gem�� einer un�bertrefflichen, von demselben
    Gentleman ausgesonnenen Kriegslist wurden auf dem Hausflur die Hunde in
    die Schw�nze gekniffen, damit sie ein w�tendes Bellen erheben m�chten,
    was sie auch taten.

    Nachdem diese Vorsichtsma�regeln getroffen waren, fa�te Mr. Giles den
    Kesselflicker fest am Arme (damit er nicht fortliefe, wie Mr. Giles
    scherzend sagte) und gab den Befehl, die T�r zu �ffnen. Brittles
    gehorchte. Einer blickte dem anderen bebend �ber die Schultern, und
    die Schar gewahrte nichts F�rchterlicheres als den armen, kleinen
    Oliver Twist, der bleich und ersch�pft die Augen aufschlug und stumm um
    Mitleid flehte.

    �Ein Knabe!� rief Mr. Giles, den Kesselflicker mutig zur�ck- und sich
    selber vordr�ngend, aus. �Was ist das -- wie -- Brittles -- erkennen
    Sie ihn?�

    Brittles, der beim �ffnen der T�r hinter dieselbe getreten war, stie�
    einen Schrei des Wiedererkennens aus, sobald er Oliver erblickte. Giles
    fa�te den Knaben bei einem Beine und einem Arme -- zum Gl�cke nicht
    bei dem verwundeten -- zog ihn herein und legte ihn der L�nge nach auf
    die Steinplatten nieder. �Hier, hier!� schrie Mr. Giles in gr��ter
    Erregtheit die Treppe hinauf, �hier ist einer von den Dieben, Ma'am!
    Wir haben einen Dieb, Mi� -- einen Verwundeten, Mi�! Ich traf ihn, Mi�,
    und Brittles hielt das Licht!�

    �In einer Laterne, Mi�!� schrie Brittles, eine Hand an den Mund
    haltend, damit sein Ruf desto sicherer hinaufdr�nge.

    Die K�chin und das Hausm�dchen liefen hinauf, um der Herrschaft
    zu verk�ndigen, da� Mr. Giles einen R�uber gefangen habe, und der
    Kesselflicker bem�hte sich, Oliver zum Bewu�tsein zur�ckzubringen,
    damit er nicht st�rbe, bevor er geh�ngt w�rde. Nach einiger Zeit
    ert�nte von oben durch den L�rm eine sanfte und wohlklingende, einer
    jungen Dame angeh�rende Stimme: �Giles, Giles!�

    �Hier, Mi�, hier bin ich! Erschrecken Sie nicht, Mi�; ich habe keinen
    bedeutenden Schaden genommen! Er leistete keinen sehr verzweifelten
    Widerstand, Mi�; ich �berw�ltigte ihn sehr bald.�

    �Still doch; Sie erschrecken ja meine Tante fast ebensosehr, wie es die
    Diebe selbst getan haben. Ist der arme Mensch stark besch�digt?�

    �Er hat eine furchtbare Wunde, Mi߻, rief Giles mit unbeschreiblichem
    Wohlbehagen hinauf.

    �Er sieht aus, als wenn er den Geist aufgeben will, Mi߻, schrie
    Brittles, wie zuvor eine Hand an den Mund haltend. �Wollen Sie nicht
    herunterkommen, Mi�, und ihn sehen, falls er --�

    �So schreien Sie doch nicht so entsetzlich. Seien Sie einen Augenblick
    still; ich will mit meiner Tante sprechen.�

    Die Sprecherin eilte mit leisen Fu�tritten fort, kehrte bald wieder
    zur�ck und erteilte den Befehl, den Verwundeten vorsichtig hinauf in
    Mr. Giles' Zimmer zu tragen; Brittles sollte sogleich den Pony satteln,
    nach Chertsey reiten und eiligst einen Konstabler und den Doktor holen.

    �Wollen Sie ihn aber nicht erst einmal sehen, Mi�?� rief Giles mit so
    viel Stolz, wie wenn Oliver ein seltener und prachtvoller Vogel w�re,
    den er heruntergeschossen h�tte.

    �Nicht um die Welt!� erwiderte die junge Dame. �Der arme Mensch! Giles,
    behandeln Sie ihn ja recht gut, und wenn es auch nur um meinetwillen
    w�re!�

    Der alte Diener des Hauses blickte, als sie sich entfernte, zu ihr
    hinauf, so stolz und wohlgef�llig, als ob sie sein eigenes Kind
    gewesen w�re, und half sodann Oliver mit der liebevollen Sorgfalt und
    Aufmerksamkeit einer Frau hinauftragen.




    29. Kapitel.

        Von den Bewohnern des Hauses, in welchem Oliver sich befand.


    In einem artigen Zimmer -- dessen Mobilien freilich mehr nach
    altmodischer Bequemlichkeit als nach moderner Eleganz aussahen --
    sa�en zwei Damen an einem wohlbesetzten Fr�hst�ckstische. Mr. Giles
    wartete im vollst�ndigen schwarzen Anzuge auf. Er stand kerzengerade
    in der Mitte zwischen dem Schenk- und Fr�hst�ckstische mit
    zur�ckgeworfenem und fast unmerklich zur Seite geneigtem Kopfe, den
    linken Fu� vorangestellt und mit der rechten Hand im Busen, w�hrend
    die herunterh�ngende Linke einen Pr�sentierteller hielt, und sah aus,
    als wenn er sich des angenehmen Bewu�tseins seiner Verdienste und
    Wichtigkeit freute.

    Die eine der beiden Damen war betagt, allein die hohe Lehne ihres
    Stuhles war nicht gerader als ihre Haltung. Ihr Anzug war ein Muster
    von Sauberkeit und Genauigkeit, altmodisch, doch nicht ohne Spuren der
    Einwirkung des Tagesgeschmacks. So sa� sie stattlich da, die gefalteten
    H�nde auf dem Tische vor ihr, die Augen -- denen die Jahre nur wenig
    von ihrem Glanze genommen -- aufmerksam auf die j�ngere Dame geheftet,
    die in der ersten zarten Bl�te der Weiblichkeit stand, eine der
    jungfr�ulichen Gestalten, von welchen wir ohne S�nde annehmen m�gen,
    da� Engel sie bewohnen, wenn der Allm�chtige zur Ausf�hrung seiner
    Absichten jemals zul��t, da� sich die Himmelsbewohner in Gestalten der
    Sterblichen verk�rpern d�rfen.

    Sie befand sich noch im siebzehnten Jahre, und ihre Figur war so leicht
    und �therisch, so zart und edel, so lieblich und sch�n, als w�re die
    Erde ihre Wohnst�tte nicht, als k�nnten die gr�beren Wesen dieser Welt
    keine zu ihr passende Mitgesch�pfe sein. Der Geist, der aus ihren
    dunkelblauen Augen leuchtete und aus ihren edlen Z�gen sprach, schien
    ihrem Alter zuvorgeeilt und kaum von dieser Welt zu sein; und doch
    verk�ndete der lebensvolle, freundlich-holde Ausdruck ihrer Mienen,
    die tausend Lichter, die auf ihrem rosigen Antlitz spielten und keinen
    Schatten auf ihm lagern lie�en, ihr L�cheln -- ihr frohes, seliges
    L�cheln -- die h�chste Gesinnungssch�ne, den reinsten Herzensadel, die
    w�rmste Liebe und Z�rtlichkeit, die besten Gef�hle und Eigenheiten der
    menschlichen Natur. Ihr L�cheln, ihr heiteres, gl�ckstrahlendes L�cheln
    war f�r h�uslichen Frieden, h�usliches Gl�ck geschaffen.

    Sie war eifrig mit den kleinen Anordnungen zum Fr�hst�ck besch�ftigt,
    und als sie zuf�llig die Augen aufschlug, w�hrend die �ltere Dame
    sie anblickte, strich sie freundlich ihr einfach auf der Stirn
    gescheiteltes Haar zur�ck, und aus ihren Blicken leuchtete eine solche
    tief-innige Z�rtlichkeit und nat�rlich-ungef�lschte Liebensw�rdigkeit
    hervor, da� selige Geister gel�chelt haben m�chten, sie so zu schauen.

    Die �ltere Dame l�chelte, doch ihr Herz war schwer, und sie trocknete
    eine Z�hre in dem freundlichen Auge ab.

    �Brittles ist also schon seit einer Stunde fort?� fragte sie nach einem
    kurzen Stillschweigen.

    �Eine Stunde und zw�lf Minuten, Ma'am�, antwortete Giles, auf seine
    silberne Uhr blickend, die er an einem schwarzen Bande herauszog.

    �Er ist immer langsam�, bemerkte die alte Dame.

    �Er war von jeher ein langsamer Bursche, Ma'am�, sagte Giles, und in
    Anbetracht dessen, da� Brittles, beil�ufig gesagt, einige drei�ig Jahre
    ein langsamer Bursche gewesen war, war es nicht eben wahrscheinlich,
    da� er jemals ein hurtiger werden w�rde.

    �Ich glaube, er wird eher schlimmer als besser�, fuhr die alte Dame
    fort.

    �Es w�rde gar nicht zu entschuldigen sein, wenn er sich aufhielte, um
    etwa mit anderen Knaben zu spielen�, fiel die junge Dame l�chelnd ein.

    Mr. Giles �berlegte offenbar, ob er sich mit Schicklichkeit auch ein
    ehrerbietiges L�cheln erlauben d�rfe, als ein Gig vorfuhr, ein dicker
    Herr heraussprang, in das Haus hereinst�rmte und so eilig in das Zimmer
    hereinpolterte, da� er fast Mr. Giles und den Teetisch umgeworfen h�tte.

    �So etwas ist mir ja in meinem ganzen Leben nicht vorgekommen!� rief
    er aus. �Meine beste Mrs. Maylie -- da� sich der Himmel erbarme --
    und obendrein in der Stille der Nacht -- es ist ganz unerh�rt, ganz
    unerh�rt!� Er sch�ttelte bei diesen Beileidsbezeigungen beiden Damen
    die H�nde, nahm Platz und erkundigte sich nach ihrem Befinden. -- �'s
    ist ein Wunder, da� der Schreck Sie nicht get�tet -- auf der Stelle
    get�tet hat!� fuhr er fort. �In aller Welt, warum schickten Sie nicht
    zu mir? Wahrhaftig, mein Bedienter h�tte in einer Minute hier sein
    sollen oder ich selbst und mein Assistent -- jedermann w�rde mit
    Freuden herbeigeeilt sein. Es versteht sich ja ganz von selbst -- unter
    solchen Umst�nden -- Himmel! -- und so unerwartet -- und in der Stille
    der Nacht!�

    Der Doktor schien besonders durch den Umstand ganz au�er sich geraten
    zu sein, da� der Einbruch unerwartet und zu n�chtlicher Zeit versucht
    worden war, als wenn es die feststehende Gewohnheit der im Fache des
    Einbrechens arbeitenden Gentlemen w�re, ihre Gesch�fte um Mittag
    abzumachen und ihr Erscheinen ein paar Tage vorher durch die Briefpost
    anzuk�ndigen.

    �Und Sie, Mi� Rose�, sagte der Doktor zu der jungen Dame; �ich --�

    �Ich befinde mich vortrefflich�, unterbrach sie ihn; �aber oben liegt
    ein Verwundeter, und die Tante w�nscht, da� Sie ihn besuchen.�

    �Ah, ich entsinne mich�, versetzte der Doktor. �Wie ich h�re, haben Sie
    ihm die Wunde beigebracht, Giles.�

    Mr. Giles, der in einem Fieber von Aufregung die Tassen geordnet hatte,
    err�tete sehr stark und erwiderte, da� er die Ehre habe.

    �Die Ehre?� sagte der Doktor. �Doch mag sein, da� es ebenso ehrenvoll
    ist, einen Dieb in einem Waschhause, wie einen Gegner auf zw�lf
    Schritte weit zu treffen. Bilden Sie sich ein, er h�tte in die Luft
    geschossen und Sie haben ein Duell gehabt, Giles.�

    Mr. Giles, der in dieser scherzhaften Behandlung der Sache einen
    ungerechten Versuch erblickte, seinen Ruhm zu verkleinern, erwiderte
    ehrerbietig, da� es seinesgleichen nicht zuk�me, ein Urteil dar�ber
    auszusprechen, allein er lebe doch des Glaubens, da� die Sache f�r den
    Getroffenen kein Spa� gewesen sei.

    �Beim Himmel, das ist wahr!� sagte der Doktor. �Wo ist er? F�hren Sie
    mich zu ihm. Ich werde bald wieder bei Ihnen sein, Mrs. Maylie. Das
    ist das kleine Fenster, durch das er eingestiegen ist? Es ist kaum zu
    glauben.�

    Er folgte, fortw�hrend sprechend, Mr. Giles die Treppe hinauf, und
    w�hrend er hinaufgeht, sei dem Leser gesagt, da� Mr. Losberne, der
    auf zehn Meilen im Umkreise unter dem Namen des �Doktors� bekannte
    Wundarzt, mehr infolge eines heiteren Temperaments als guten Lebens
    beleibt geworden und ein so gutherziger und biederer, nebenher auch
    wunderlicher alter Junggeselle war, da� man in einem f�nfmal gr��eren
    Umkreise kaum seinesgleichen finden d�rfte.

    Der Doktor blieb weit l�nger fort, als es die Damen vermutet hatten. Es
    wurde ein langer, flacher Kasten aus dem Gig geholt, h�ufig geklingelt,
    die Dienerschaft lief treppauf, treppab; mit einem Worte, es mu�te
    wohl etwas Wichtiges vorgehen. Endlich trat er mit einer �u�erst
    geheimnisvollen Miene wieder herein, verschlo� die T�r sorgf�ltig und
    sagte, w�hrend er mit dem R�cken an sie gelehnt stehenblieb, als wenn
    er verhindern wollte, da� jemand hereink�me: �Mrs. Maylie, dies ist ein
    ganz wunderbarer Fall.�

    �Ich will doch hoffen, da� der Patient sich nicht in Gefahr befindet?�
    fragte die alte Dame.

    �Es w�rde den Umst�nden nach nicht zu verwundern sein,� erwiderte
    Losberne, �obwohl ich es nicht glaube. Haben Sie den Dieb gesehen?�

    �Nein.�

    �Auch sich ihn nicht beschreiben lassen?�

    �Nein.�

    �Bitt' um Vergebung, Ma'am�, fiel Giles ein; �ich wollte Ihnen eben
    eine Beschreibung von ihm geben, als Doktor Losberne erschien.�

    Die Sache verhielt sich indes so, da� sich Mr. Giles nicht hatte
    �berwinden k�nnen, das Gest�ndnis zu machen, da� er nur einen Knaben
    getroffen habe. Er hatte wegen seines mutvollen Benehmens so gro�e
    Lobspr�che erhalten, da� er nicht umhin gekonnt, die Aufhellung der
    Sache noch ein paar entz�ckende Minuten aufzuschieben, um noch ein
    Weilchen in dem s��en Bewu�tsein des Ruhmes einer unersch�tterlichen
    Herzhaftigkeit zu schwelgen.

    �Rose w�nschte den Mann zu sehen,� sagte Mrs. Maylie, �allein ich
    wollte nichts davon h�ren.�

    �Hm!� sagte der Doktor. �Er sieht aber nicht eben sehr f�rchterlich
    aus. M�chten Sie ihn auch nicht in meiner Gegenwart sehen?�

    �Warum nicht, wenn Sie es f�r notwendig halten?� erwiderte die alte
    Dame.

    �Ich mu� es f�r notwendig erkl�ren oder bin doch jedenfalls �berzeugt,
    da� Sie es gar sehr bedauern w�rden, es nicht getan zu haben, wenn Sie
    ihn sp�ter zu sehen bek�men. Er ist vollkommen ruhig, und wir haben
    auch in allen Beziehungen f�r ihn gesorgt. Erlauben Sie mir Ihren Arm,
    Mi� Rose. Auf meine Ehre, Sie brauchen nicht im mindesten Furcht zu
    hegen.�




    30. Kapitel.

        Was die beiden Damen Maylie und Doktor Losberne von Oliver denken.


    Der Doktor legte unter noch viel anderen redseligen Versicherungen, da�
    die Damen durch den Anblick des Verbrechers angenehm �berrascht werden
    w�rden, den Arm der j�ngeren in den seinigen, bot Mrs. Maylie seine
    andere freie Hand und f�hrte sie mit der f�rmlichsten Galanterie die
    Treppe hinauf.

    �Lassen Sie mich nun h�ren, was Sie von ihm denken�, sagte er, als sie
    vor der T�r des Patienten standen. �Er hat sich seit vielen Tagen den
    Bart nicht abnehmen lassen, sieht aber trotzdem keineswegs wie ein
    Gurgelabschneider aus.�

    Er f�hrte die Damen hinein und an das Bett, schob die Vorh�nge zur�ck,
    und sie erblickten statt eines grimmig aussehenden Banditen, den
    sie zu sehen erwartet hatten -- einen vor Schmerz und Ersch�pfung
    eingeschlafenen Knaben. Olivers verbundener Arm lag auf seiner Brust,
    und sein Kopf ruhte auf dem andern, der durch sein langes, wallendes
    Haar fast versteckt war. Rose setzte sich, w�hrend Losberne im
    Anschauen des Knaben verloren dastand, oben an das Bett des letzteren,
    beugte sich �ber ihn und strich ihm leise das Haar von der Stirn, auf
    welche ein paar Tr�nen aus ihrem Auge herabfielen.

    Der Knabe bewegte sich und l�chelte im Schlafe, als wenn ihn diese
    Zeichen des Mitgef�hls und zarten Erbarmens in einen s��en Traum
    von nie gekannter Liebe und Z�rtlichkeit versenkt h�tten, so wie
    entfernte T�ne einer lieblichen Melodie oder das Rauschen des Wassers
    an einem heimlichen Pl�tzchen oder der Duft einer Blume oder selbst
    das Aussprechen eines teuren Namens bisweilen pl�tzlich unbestimmte
    Bilder in diesem Dasein nie erlebter Szenen, die gleich einem Hauche
    wieder verschwinden, vor die Seele zaubert, Szenen, die aus der dunklen
    Erinnerung eines l�ngst vergangenen gl�cklichen Daseins emporzutauchen
    scheinen, denn keine Kraft der menschlichen Seele vermag es, sie wieder
    zur�ckzurufen.

    �Ich bin fast au�er mir vor Verwunderung�, fl�sterte die alte Dame.
    �Dieses arme Kind kann nun und nimmermehr ein Diebes- und R�uberz�gling
    sein.�

    �Das Laster schl�gt seinen Wohnsitz in gar vielerlei Tempeln auf�,
    versetzte Losberne seufzend, indem er den Vorhang wieder fallen lie�,
    �und erscheint oft genug in lieblicher Gestalt.�

    �Aber doch nicht bei solcher Jugend�, fiel Rose ein.

    �Meine teure Mi�,� entgegnete der Wundarzt mit traurigem Kopfsch�tteln,
    �das Verbrechen beschr�nkt sich gleich dem Tode nicht auf die Bejahrten
    und Abgelebten allein. Die Jugendlichsten und Sch�nsten sind nur zu oft
    seine auserw�hlten Opfer.�

    �O Sir, k�nnen Sie wirklich glauben, da� dieser zarte Knabe sich
    freiwillig den schlimmsten B�sewichtern zugesellt hat?� wandte Rose
    lebhaft ein.

    Losberne sch�ttelte den Kopf mit einer Miene, als ob er es f�r sehr
    m�glich hielte, und f�hrte die Damen in das ansto�ende Zimmer, damit
    der kleine Patient, wie er sagte, nicht gest�rt w�rde.

    �Aber selbst, wenn er ruchlos gewesen w�re,� fuhr Rose fort, �so
    bedenken Sie, wie jung er ist; da� er vielleicht nie eine liebevolle
    Mutter, vielleicht nicht einmal ein elterliches Haus gekannt hat, und
    wie wahrscheinlich es ist, da� ihn schlechte Behandlung, Schl�ge oder
    Hunger gen�tigt haben, sich an Menschen anzuschlie�en, die ihn zum
    Verbrechen zwangen. Tante, beste Tante, bedenken Sie das doch ja, ehe
    Sie zugeben, da� der kranke Kleine in ein Gef�ngnis geschleppt wird,
    das auf alle F�lle das Grab jeder Hoffnung der Besserung bei ihm sein
    w�rde. Oh, so gewi� Sie mich liebhaben und wissen, da� ich bei Ihrer
    G�te und Z�rtlichkeit meine Elternlosigkeit nie empfunden, da� ich sie
    aber schmerzlich h�tte f�hlen k�nnen und gleich hilf- und schutzlos wie
    dies arme Kind sein k�nnte, haben Sie Mitleid mit ihm, ehe es zu sp�t
    ist.�

    �Mein liebes Kind,� sagte die �ltere Dame, das weinende M�dchen an
    die Brust dr�ckend, �glaubst du, ich w�rde auch nur ein Haar seines
    Hauptes kr�mmen lassen wollen?�

    �O nein, nein, beste Tante, Sie wollen und k�nnten es nicht!� rief Rose
    mit Lebhaftigkeit aus.

    �Nein, sicherlich nicht,� fuhr Mrs. Maylie mit bebender Lippe fort,
    �meine Tage neigen sich ihrem Ende zu, und m�ge ich Barmherzigkeit
    erfahren, wie ich sie anderen erweise. Was kann ich zur Rettung des
    Knaben tun, Sir?�

    �Lassen Sie mich nachdenken, Ma'am,� erwiderte der Doktor, �lassen Sie
    mich nachdenken.�

    Mr. Losberne steckte seine H�nde in die Taschen und ging einigemal im
    Zimmer auf und nieder, stand dann wieder still, wiegte sich auf seinen
    Fu�spitzen, rieb heftig die Stirn und sagte endlich: �Ich hab's, Ma'am.
    -- Ja -- ich sollte meinen, da� ich es schon einrichten k�nnte, wenn
    Sie mir unbeschr�nkte Vollmacht geben wollen, Giles und Brittles, den
    gro�en Jungen, in das Bockshorn zu jagen. Giles ist ein alter Diener
    Ihres Hauses und ein treuer Mensch, das wei� ich; und Sie k�nnen es
    bei ihm auf tausenderleiweise wieder gutmachen und ihn obendrein daf�r
    belohnen, da� er ein so guter Sch�tze ist. Sie haben doch nichts
    dawider?�

    �Wenn es kein anderes Mittel gibt, das Kind zu retten, nein�,
    antwortete Mrs. Maylie.

    �Auf mein Wort, es gibt kein anderes Mittel�, versicherte Losberne.

    �Dann bekleidet Tante Sie mit Vollmacht�, sagte Rose, durch Tr�nen
    l�chelnd; �aber bitte, setzen Sie den beiden guten Leuten nicht h�rter
    zu, als es unumg�nglich notwendig ist.�

    �Sie scheinen zu glauben,� entgegnete der Doktor, �da� alle Welt heute
    zu Hartherzigkeit geneigt ist, Sie selbst allein ausgenommen, Mi�
    Rose. Ich will nur um des aufwachsenden m�nnlichen Geschlechts willen
    insgeheim hoffen, da� der erste Ihrer w�rdige junge Mann, der Ihr
    Mitleid in Anspruch nimmt, seine Werbung bei Ihnen anbringt, wenn Sie
    sich in einer ebenso verwundbaren und weichherzigen Stimmung befinden,
    und w�nschte nichts mehr, als da� ich selbst ein junges Herrlein sein
    m�chte, um sogleich einen so g�nstigen Augenblick wie den gegenw�rtigen
    benutzen zu k�nnen.�

    �Sie sind ein ebenso gro�er Knabe wie unser guter Brittles�, sagte Rose
    err�tend.

    �Dazu geh�rt eben nicht viel�, versetzte der Doktor herzlich lachend.
    �Doch um auf den kleinen Knaben zur�ckzukommen: wir haben die
    Hauptsache bei unserem Vertrage noch nicht erw�hnt. Er wird ohne
    Zweifel in ungef�hr einer Stunde aufwachen, und obgleich ich dem
    breitm�uligen Konstabler unten gesagt habe, da� bei Gefahr seines
    Lebens nicht mit ihm gesprochen werden d�rfe, so denke ich doch, da�
    wir es ganz dreist tun k�nnen. Ich mache nun die Bedingung -- da� ich
    ihn in Ihrer Gegenwart examiniere, und da� er, wenn wir seinen Aussagen
    nach urteilen m�ssen, und wenn ich Ihnen zur Befriedigung Ihres kalten
    Verstandes dartun kann, da� er (was mehr als m�glich) durch und durch
    verderbt ist, seinem Schicksale ohne weitere Einmischung -- zum
    wenigsten von meiner Seite -- �berlassen wird.�

    �Nein, Tante, nein!� flehte Rose.

    �Ja, Tante, ja!� sagte der Doktor. �Sind wir einig?�

    �Er kann nicht im Laster verh�rtet sein�, sagte Rose; �es ist
    unm�glich.�

    �Desto besser�, entgegnete Losberne; �dann ist um so mehr Grund
    vorhanden, meinen Vorschlag dreist anzunehmen.�

    Der Vertrag wurde endlich geschlossen, und man setzte sich, um in
    gro�er Spannung Olivers Erwachen abzuwarten.

    Die Geduld der beiden Damen sollte indes auf eine l�ngere Probe
    gestellt werden, als sie nach des Doktors �u�erungen gef�rchtet hatten,
    denn eine Stunde verging nach der andern, und Oliver lag fortw�hrend im
    festesten Schlummer. Es war Abend geworden, als ihnen der gutherzige
    Losberne die Nachricht brachte, da� der Patient endlich hinreichend
    wach geworden sei, um Rede und Antwort stehen zu k�nnen. Er sei sehr
    krank, wie Losberne sagte, und sehr schwach infolge des Blutverlustes,
    allein sein Gem�t, durch den Wunsch, etwas zu enth�llen, so beunruhigt,
    da� es unbedingt besser sei, ihn reden zu lassen, als -- was sonst
    geschehen sein w�rde -- darauf zu bestehen, da� er sich bis zum
    folgenden Morgen ruhig verhalten solle.

    Die Unterredung dauerte lange, denn Oliver erz�hlte ihnen seine ganze
    Lebensgeschichte, und oft n�tigten ihn Schmerz oder Ersch�pfung,
    innezuhalten. Die schwache Stimme des kranken Knaben, sein r�hrend
    schauerlicher Bericht �ber eine lange Reihe trostloser Leiden und
    Mi�geschicke, von verh�rteten Menschen �ber ihn verh�ngt, h�rte sich in
    dem verdunkelten Zimmer gar feierlich an. Oh, wieviel weniger Unrecht
    und Ungerechtigkeit, Leid und Gr�men, Grausamkeit und Elend, wie es
    jeder Tag mit sich bringt, w�rde es auf dieser Welt geben, wenn wir --
    w�hrend wir unsere Mitmenschen unterdr�cken und qu�len -- nur mit einem
    einzigen Gedanken an die finster drohenden Anklagen gegen uns d�chten,
    die gleich dichten, schweren Wolken freilich langsam, aber desto
    gewisser zum Himmel emporsteigen, um dereinst ihre Racheblitze auf
    unsere H�upter herabzusenden -- wenn wir im Geist nur einen Augenblick
    h�ren wollten auf das schauerliche Zeugnis der Stimmen der Toten und
    zu ihrem und unserem Sch�pfer und Richter Hin�bergegangenen, die keine
    menschliche Macht oder Gewalt unterdr�cken, kein Stolz verstummen
    machen kann!

    Olivers Kissen war in dieser Nacht durch Frauenh�nde gegl�ttet, und
    Liebensw�rdigkeit und Tugend bewachten seinen Schlummer. Er empfand
    eine selige Ruhe und h�tte sterben m�gen ohne Murren.

    Sobald die Unterredung mit ihm beendet und er, was fast augenblicklich
    geschah, wieder eingeschlummert war, trocknete der Doktor seine Augen,
    verw�nschte sie wie gew�hnlich wegen ihrer Schw�che und begab sich
    darauf in die K�che hinunter, um seinen Feldzug gegen Mr. Giles und
    Konsorten zu beginnen. Er fand die ganze Dienerschaft nebst dem
    Konstabler und dem Kesselflicker versammelt, der in Anbetracht seiner
    geleisteten Dienste eine besondere Einladung erhalten hatte, den
    ganzen Tag zu bleiben und sich wieder zu st�rken und zu erquicken. Der
    Konstabler war ein Gentleman mit einem gro�en Stabe, gro�em Kopfe,
    gro�em Munde und gro�en Halbstiefeln und sah aus, als wenn er sehr
    reichlich im gespendeten Ale gezecht h�tte, was auch in der Tat der
    Fall war. Als der Doktor eintrat, wurden noch immer die Abenteuer der
    vergangenen Nacht besprochen, Mr. Giles verbreitete sich �ber seine
    Geistesgegenwart, und Brittles bekr�ftigte, mit einem Alekrug in der
    Hand, alles, was Mr. Giles erst noch sagen wollte.

    �Bleibt sitzen�, sagte der Doktor mit einer Handbewegung.

    �Sch�nen Dank, Sir�, sagte Mr. Giles. �Misses befahlen mir, ein wenig
    Ale auszuteilen, und da es mir in meinem eignen kleinen Zimmer zu eng
    war, und da mich nach Gesellschaft verlangte, so trinke ich meinen
    Anteil hier.�

    Brittles und die �brigen dr�ckten durch ein leises Gemurmel ihr
    Vergn�gen �ber Mr. Giles' Herablassung aus, und Mr. Giles blickte mit
    einer G�nnermiene umher, welche deutlich sagte, da� er, solange sie
    ein schickliches Benehmen beobachteten, ihre Gesellschaft sicher nicht
    verlassen w�rde.

    �Wie befindet sich der Patient heute abend, Sir?� fragte Giles.

    �Nicht eben gar zu gut�, erwiderte der Doktor. �Ich f�rchte, Giles, da�
    Sie sich selbst in eine arge Klemme gebracht haben.�

    �Ich will doch hoffen, Sir, Sie wollen nicht sagen, da� er sterben
    werde�, sagte Giles zitternd. �Ich k�nnte nie wieder ruhig werden, wenn
    es gesch�he. Sir, ich m�chte um alles Silberzeug im Lande keinem Knaben
    das Leben nehmen, nicht einmal Brittles.�

    �Das ist nicht der Kernpunkt der Sache�, fuhr der Doktor geheimnisvoll
    fort. �F�rchten Sie Gott, und haben Sie ein Gewissen, Giles?�

    �Ja, Sir, ich sollte meinen�, stotterte der sehr bla� gewordene
    Haushofmeister.

    �Und wie steht es mit Ihnen, junger Mensch -- haben Sie auch ein
    Gewissen, Brittles?�

    �Barmherziger Himmel, Sir -- wenn Mr. Giles ein Gewissen hat, hab' ich
    auch eins.�

    �Dann sagt mir beide -- alle beide: wollt ihr es auf euer Gewissen
    nehmen, zu beschw�ren, da� der verwundete, oben liegende Knabe derselbe
    ist, der gestern nacht durch das kleine Fenster gesteckt wurde? Heraus
    mit der Sprache! Sagt an, sagt an!�

    Der Doktor, der aller Welt als der sanftm�tigste Mann von der Welt
    bekannt war, sprach diese Worte in einem so schauerlich-zornigen Tone,
    da� Giles und Brittles, die durch Ale und Aufregung ziemlich au�er
    Fassung waren, einander vollkommen bet�ubt anstarrten. -- �Achten Sie
    auf die Antwort, welche erfolgen wird, Konstabler�, sprach der Doktor
    weiter und hob mit gro�er Feierlichkeit den Zeigefinger empor; �es kann
    fr�her oder sp�ter viel darauf ankommen.�

    Der Konstabler nahm eine so weise Miene an, wie er nur konnte, und
    griff zu seinem Stabe.

    �Sie werden bemerken, es handelt sich einfach um die Identit�t der
    Person�, fuhr der Doktor fort.

    �Sie haben vollkommen recht, Sir�, sagte der Konstabler unter heftigem
    Husten, denn er hatte rasch seinen Krug geleert, wovon ihm etwas in die
    unrechte Kehle gekommen war.

    �Es wird in das Haus eingebrochen,� sagte der Doktor, �und zwei Leute
    sehen einen Knaben auf einen einzigen fl�chtigen Augenblick, mitten
    im Pulverdampfe, in der Verwirrung des n�chtlichen Schreckens und
    Aufruhrs. Am folgenden Morgen kommt ein Knabe in dieses Haus, und weil
    er zuf�llig den Arm verbunden hat, legen die Leute gewaltsam Hand an
    ihn, bringen sein Leben dadurch in die augenscheinlichste Gefahr und
    schw�ren, da� er an dem Einbruch teilgenommen habe. Die Frage ist nun
    die, ob das Verhalten besagter Leute durch die Umst�nde gerechtfertigt
    erscheint, und wo nicht, in was f�r eine Lage sie sich selber
    versetzen? Und nun noch einmal,� donnerte der Doktor, w�hrend der
    Konstabler Giles und Brittles mit bedenklich-mitleidiger Miene ansah,
    �seid ihr gewillt und imstande, vor Gott und auf das heilige Evangelium
    die Identit�t des Knaben zu beschw�ren?�

    Brittles blickte Giles und Giles Brittles zweifelhaft und fragend an;
    der Konstabler hielt die Hand hinter das Ohr, damit ihm ja nichts
    von der Antwort entgehen m�chte; die K�chin, das Hausm�dchen und der
    Kesselflicker beugten sich vor, um zu lauschen, und der Doktor schaute
    mit scharfen Blicken umher, als das Heranrollen eines Wagens und gleich
    darauf das L�uten der Gartentorglocke vernommen wurde.

    �Es sind die Polizeim�nner aus London�, rief Brittles, sich sehr
    erleichtert f�hlend, aus.

    �In aller Welt, wie kommen denn die hierher?� fragte der Doktor,
    seinerseits erschreckend.

    �Ich und Mr. Giles haben heute morgen nach ihnen geschickt,� antwortete
    Brittles, �und ich wundere mich nur, da� sie so sp�t kommen.�

    �Ah, Sie schickten nach ihnen! Ei, so wollt' ich, da� dieser und jener
    Sie holte! Ihr seid hier doch lauter verw�nschte Dummk�pfe!� sagte der
    Doktor im Hinauseilen.




    31. Kapitel.

        Eine kritische Situation.


    �Wer ist hier?� fragte Brittles, indem er die Haust�r ein wenig �ffnete
    und die Kerze mit der Hand beschattend, hinausschaute.

    ��ffnen Sie die T�r�, entgegnete ein Mann von drau�en. �Es sind die
    Polizeibeamten aus Bow-Street, nach denen heut' geschickt worden ist.�

    Durch diese Auskunft v�llig beruhigt, �ffnete Brittles die T�r in ihrer
    vollen Breite und stand einem stattlichen Manne in einem gro�en Mantel
    gegen�ber, der sofort ohne weiteres eintrat und sich die Stiefel so
    ruhig auf der Matte reinigte, als geh�re er ins Haus.

    �Schicken Sie sofort jemand, der meinem Kollegen die Sorge f�r Pferd
    und Wagen abnimmt. Haben Sie nicht eine Remise hier, da� wir den Wagen
    kurze Zeit unterstellen k�nnen?�

    Als Brittles eine bejahende Antwort gab und auf das Geb�ude deutete,
    schritt der stattliche Mann zur Gartenpforte zur�ck und half seinem
    Kollegen beim Aussteigen aus dem Gig, wobei ihnen Brittles mit dem
    Ausdruck hoher Bewunderung leuchtete. Hierauf kehrten beide Beamte nach
    dem Hause zur�ck und legten, ins Besuchszimmer geleitet, ohne weiteres
    �berrock und Hut ab. Der erste, der geklopft hatte, war ein starker
    Mann von Mittelgr��e, etwa f�nfzig Jahre alt, und hatte gl�nzendes,
    ziemlich kurz geschnittenes Haar, ein rundes Gesicht und scharfe Augen.
    Der andere war ein Rotkopf und hager, trug Stulpenstiefel und hatte ein
    absto�endes Gesicht und eine aufgeworfene, widerw�rtige Nase.

    �Melden Sie Ihrer Herrschaft, da� Blathers und Duff hier w�ren�, sagte
    der stattlichere von beiden, sein Haar niederstreichend und ein Paar
    Handfesseln auf den Tisch legend. �Ah! guten Abend, Sir. Kann ich ein
    W�rtchen allein mit Ihnen reden?�

    Diese Anrede war an Mr. Losberne gerichtet, der eben mit den beiden
    Damen eintrat und Brittles einen Wink gab, hinauszugehen. �Dies ist die
    Dame des Hauses�, sagte Losberne mit einer Handbewegung auf Mrs. Maylie
    zu.

    Mr. Blathers machte eine Verbeugung. Auf die Aufforderung, Platz
    zu nehmen, stellte er seinen Hut auf den Fu�boden, setzte sich und
    veranla�te Duff, das gleiche zu tun. Der letztere, der sich weniger in
    guter Gesellschaft bewegt zu haben schien oder sich doch jedenfalls
    nicht mit gro�er Leichtigkeit darin bewegte, nahm erst nach manchem
    umst�ndlichen Kratzfu�e Platz und legte dann sofort den Knauf seines
    Handstockes an den Mund.

    �Lassen Sie uns nun aber sogleich auf den hier ver�bten Einbruch
    kommen, Sir�, sagte Blathers. �Wie verh�lt es sich mit der Sache?�

    Losberne w�nschte Zeit zu gewinnen und berichtete der L�nge nach und
    mit gro�er Weitschweifigkeit. Die Herren Blathers und Duff h�rten mit
    �u�erst weisen Mienen zu und blinzelten einander dann und wann sehr
    pfiffig zu.

    �Ich kann �ber die Sache nat�rlich nicht eher etwas Gewisses sagen,�
    bemerkte Blathers, als Losberne mit seinem Bericht zu Ende war, �als
    bis ich die Stelle in Augenschein genommen habe, wo der Einbruch
    versucht worden ist; jedoch meine Meinung ist rund heraus die -- denn
    ich stehe, selbst auf die Gefahr, zu irren, nicht an, so weit zu gehen
    -- da� er von keinem Kaffer ver�bt ist -- was sagst du, Duff?�

    Duff war derselben Meinung.

    �Sie wollen sagen,� versetzte Losberne l�chelnd, �der Einbruch sei von
    keinem Landmanne, von keinem Nicht-Londoner ver�bt?�

    �Ganz recht, Sir. Wissen Sie noch etwas �ber das Verbrechen zu sagen?�

    Losberne verneinte.

    �Was ist denn das aber mit dem Knaben, von dem die Dienerschaft im
    Hause spricht?�

    �O ganz und gar nichts�, erwiderte der Doktor. �Der Haushofmeister
    hatte es sich in seiner Best�rzung in den Kopf gesetzt, der Knabe w�re
    bei dem Einbruche, der Himmel wei� wie, beteiligt gewesen -- 's ist
    aber durchaus nichts als Torheit und alberne Einbildung gewesen.�

    �Das hei�t die Sache gar zu sehr auf die leichte Achsel nehmen�,
    bemerkte Duff.

    �Du hast ganz recht, Duff�, sagte Blathers mit bekr�ftigendem
    Kopfnicken und mit den Handfesseln spielend, als wenn sie ein Paar
    Kastagnetten gewesen w�ren. �Wer ist der Knabe? Welche Auskunft gibt er
    �ber sich? Woher kam er? Er wird doch nicht aus den Wolken gefallen
    sein, Sir?�

    �Nat�rlich, nein�, sagte Losberne, den Damen einen unruhigen Blick
    zuwerfend. �Mir ist indessen sein ganzer Lebenslauf bekannt, und --
    doch wir k�nnen nachher dar�ber sprechen. Wollen Sie nicht vor allen
    Dingen die Stelle sehen, wo die Diebe einzubrechen versuchten?�

    �Allerdings,� erwiderte Blathers. �Wir nehmen zuerst die Stelle in
    Augenschein und verh�ren sodann die Dienerschaft. Das pflegt der
    gew�hnliche Gang des Gesch�fts zu sein.�

    Es wurde Licht gebracht, und die Herren Blathers und Duff, in
    Begleitung des Konstablers des Ortes, Brittles', Giles' und, mit einem
    Worte, s�mtlicher sonstiger Hausbewohner, begaben sich in das kleine
    Gemach am Ende des Hausflurs und sahen aus dem Fenster, gingen darauf
    hinaus und sahen in das Fenster hinein, besichtigten den Fensterladen,
    sp�rten den Fu�tritten nach beim Scheine einer Laterne und durchstachen
    die B�sche vermittels einer Heugabel. Nachdem dies alles geschehen war
    und alle das Vorgehen der Beamten mit atemloser Teilnahme verfolgt
    hatten, gingen Blathers und Duff wieder hinein und vernahmen Giles und
    Brittles �ber ihren Anteil an den Begebenheiten der Schreckensnacht;
    beide Diener erz�hlten sechsmal statt einmal und widersprachen einander
    beim ersten nur in einem einzigen wichtigen Punkte und beim letzten nur
    in einem Dutzend wesentlicher Aussagen. Nach Beendigung des Verh�rs
    wurden Giles und Brittles entlassen, und die Herren Blathers und Duff
    hielten eine lange Beratung ab, im Vergleich zu der in Beziehung auf
    Heimlichkeit und Feierlichkeit eine Konsultation ber�hmter Doktoren
    �ber den schwierigsten Krankheitsfall blo�es Kinderspiel gewesen w�re.

    Losberne ging unterdessen im ansto�enden Zimmer sehr unruhig auf und
    ab, und Mrs. Maylie und Rose schauten ihm mit noch gr��erer Unruhe zu.

    �Auf mein Wort,� sagte er, pl�tzlich stillstehend, �ich wei� kaum, was
    hier zu tun ist.�

    �Wenn den beiden M�nnern�, versetzte Rose, �die Geschichte des
    ungl�cklichen Knaben erz�hlt w�rde, wie sie ist, es w�re sicher genug,
    ihn in ihren Augen von Schuld zu entlasten.�

    �Das mu� ich bezweifeln, meine werte junge Dame�, wandte der Doktor
    kopfsch�ttelnd ein. �Ich glaube nicht, da� es sie oder auch die h�heren
    Polizei- oder Justizbeamten befriedigen w�rde. Sie w�rden sagen, er sei
    jedenfalls ein fortgelaufener Kirchspielknabe und Lehrling. Nach rein
    weltlich-verst�ndigen Erw�gungen und Wahrscheinlichkeiten beurteilt,
    unterliegt seine Geschichte gro�en Zweifeln.�

    �Sie schenken ihr doch Glauben?� fiel Rose hastig ein.

    �Ich schenke ihr Glauben, so befremdlich sie lautet, und bin vielleicht
    ein gro�er Tor, weil ich es tue,� versetzte der Doktor; �allein
    nichtsdestoweniger halte ich sie keineswegs f�r eine solche, die einen
    erfahrenen Polizeibeamten zufriedenstellen w�rde.�

    �Warum aber nicht?� fragte Rose.

    �Meine sch�ne Inquirentin,� erwiderte Losberne, �weil in ihr, wenn
    man sie mit den Augen jener Herren betrachtet, so viele b�se Umst�nde
    vorkommen. Der Knabe kann nur beweisen, was �bel, und nichts von dem,
    was gut aussieht. Die verw�nschten Sp�rhunde werden nach dem Warum
    und Weshalb fragen und nichts als wahr gelten lassen, was ihnen nicht
    vollst�ndig bewiesen wird. Er sagt selbst, da� er sich eine Zeitlang in
    der Gesellschaft von Diebesgelichter befunden, eines Taschendiebstahls
    angeklagt vor einem Polizeiamte gestanden hat, und aus dem Hause
    des bestohlenen Herrn gewaltsam entf�hrt ist, er kann selbst nicht
    angeben, hat nicht einmal eine Vermutung, wohin. Er wird von M�nnern
    nach Chertsey hergebracht, die ganz vernarrt in ihn zu sein scheinen
    und ihn durch ein Fenster stecken, um ein Haus zu pl�ndern; und gerade
    in dem Augenblicke, wo er die Bewohner wecken und tun will, was
    seine Unschuld ins Licht setzen w�rde, verrennt ihm der verw�nschte
    Haushofmeister den Weg und schie�t ihn in den Arm, gleichsam recht
    absichtlich, um ihn daran zu hindern, etwas zu tun, das ihm n�tzen
    k�nnte. Leuchtet Ihnen das alles nicht ein?�

    �Nat�rlich leuchtet es mir ein�, erwiderte Rose, den Eifer des Doktors
    bel�chelnd; �allein ich sehe nur noch immer nichts darin, wodurch die
    Schuld des armen Kindes erwiesen w�rde.�

    �Nicht -- ei!� rief Losberne aus. �O, �ber die hellen, scharfen
    �ugelein der Damen, womit sie, sei es zum Guten oder B�sen, immer nur
    die eine Seite an einer Sache oder Frage sehen, und zwar stets die, die
    sich ihnen eben zuerst dargeboten hat!�

    Nachdem er seinem Herzen Luft dadurch gemacht, da� er Mi� Rose diesen
    Erfahrungssatz zu Gem�t gef�hrt, steckte er die H�nde in die Taschen
    und fing wieder an, mit noch rascheren Schritten als zuvor im Zimmer
    auf und ab zu gehen. �Je mehr ich dar�ber nachdenke,� fuhr er fort,
    �desto zahlreichere und gr��ere Schwierigkeiten sehe ich voraus, den
    beiden Leuten die Geschichte des Knaben glaubhaft zu machen. Ich bin
    �berzeugt, da� sie ihm schlechterdings keinen Glauben schenken werden,
    und selbst wenn sie ihm am Ende nichts anhaben k�nnen, so werden doch
    ihre Zweifel und der Verdacht, den diese wieder auf ihn werfen m�ssen,
    von sehr wesentlichem Nachteile f�r den wohlwollenden Plan sein, ihn
    aus dem Elende zu retten.�

    �O bester Doktor, was ist da zu tun?� rief Rose aus. �Du lieber Himmel,
    da� Giles auch den unseligen Einfall hat haben m�ssen, nach der Polizei
    zu schicken!�

    �Ich w��te nicht, was ich darum g�be, wenn es nicht geschehen w�re�,
    fiel Mrs. Maylie ein.

    �Ich wei� nur eins,� sagte der Doktor, sich mit einer Art Ruhe
    der Verzweiflung hinsetzend, �da� ich die Kerle mit g�ttlicher
    Unversch�mtheit aus dem Hause zu bringen suchen mu�. Der Zweck ist ein
    guter, und darin liegt die Entschuldigung. Bei dem Knaben zeigen sich
    starke Fiebersymptome, und er befindet sich in einem Zustande, da� er
    f�r jetzt nicht mehr befragt werden darf; das ist ein Trost. Wir m�ssen
    seine Lage so gut wie m�glich zu benutzen suchen, und wenn es nicht
    gl�cken will, so ist es nicht unsere Schuld. Herein!�

    Blathers und Duff erschienen, und der erstere sprach sogleich ein
    Urteil �ber den Einbruch in einem Kauderwelsch aus, das weder Losberne
    noch die Damen verstanden. Um eine Erkl�rung gebeten, sagte er, dem
    Doktor einen ver�chtlichen Blick zuwerfend und sich mitleidig zu den
    Damen wendend, er meine, da� die Dienerschaft bei dem beabsichtigten
    Raube nicht im Einverst�ndnisse gewesen sei.

    �Wir haben auch durchaus keinen Verdacht gegen sie gehabt�, bemerkte
    Mrs. Maylie.

    �Mag wohl sein, Ma'am�, entgegnete Blathers; �sie konnte aber auch Hand
    im Spiele gehabt haben.�

    �Und eben weil kein Verdacht sie traf,� fiel Duff ein, �mu�te um so
    mehr danach geforscht werden.�

    �Wir haben gefunden, da� der Einbruch Londoner Werk ist�, fuhr Blathers
    fort; �die Kerle haben meisterhaft gearbeitet.�

    �In Wahrheit sehr wackere Arbeit�, bemerkte Duff leise.

    �Der Einbrecher sind zwei gewesen,� berichtete Blathers weiter, �und
    sie haben einen Knaben bei sich gehabt, was aus der Gr��e des Fensters
    klar ist. Mehr l��t sich f�r jetzt nicht sagen. Zeigen Sie uns doch den
    Burschen, den Sie im Hause haben.�

    �Die Herren nehmen aber wohl erst ein wenig zu trinken an, Mrs.
    Maylie�, sagte der Doktor mit erheiterten Mienen, als wenn ihm ein
    neuer Gedanke aufgegangen w�re.

    �Gewi߻, fiel Rose eifrig ein. �Es steht Ihnen sogleich alles zu
    Diensten, wenn Sie befehlen.�

    �Besten Dank, Mi߻, sagte Blathers, mit dem Rock�rmel �ber den Mund
    fahrend. �So ein Verh�r ist trockene Arbeit. Was Sie eben zur Hand
    haben, Mi�; machen Sie sich unsertwegen keine Ungelegenheiten.�

    �Was belieben Sie?� fragte der Doktor, der jungen Dame nach dem
    Eckschrank folgend.

    �Wenn's Ihnen gleichviel ist, 'nen Tropfen Branntwein, Sir�, erwiderte
    Blathers. �Wir hatten 'ne kalte Fahrt von London her, und der
    Branntwein l�uft einem so warm �bers Herz.�

    Er richtete die letzteren Worte an Mrs. Maylie, und der Doktor
    schl�pfte unterdes aus dem Zimmer.

    �Ah, meine Damen,� fuhr Blathers, das ihm gereichte Glas vor das Auge
    emporhaltend, fort, �ich habe in meinem Leben die schwere Menge solcher
    Geschichten erlebt.�

    �Zum Beispiel den Einbruch in Edmonton, Blathers�, fiel Duff ein.

    �Ja, ja�, sagte Blathers; �der war diesem allerdings �hnlich genug. Er
    wurde von dem Conkey Chickweed begangen.�

    �Das hast du immer behauptet�, entgegnete Duff; �aber ich sage dir,
    die Familie Pet hat ihn ver�bt, und Conkey hat nicht mehr die Hand im
    Spiele dabei gehabt als ich.�

    �Ei was,� sagte Blathers, �ich wei� es besser. Entsinnst du dich noch,
    wie sich Conkey sein Geld stehlen lie�? Es war 'ne Geschichte, noch
    merkw�rdiger, als sie in 'nem Buche vorkommen kann.�

    �Erz�hlen Sie doch�, nahm Rose das Wort, um die unwillkommenen G�ste
    bei guter Laune zu erhalten.

    �Es war 'ne Spitzb�berei, worauf so leicht niemand verfallen sein
    w�rde, Mi߻, begann Blathers. �N�mlich der Conkey Chickweed --�

    �Conkey bedeutet soviel als Emmesgatsche[AN], Ma'am�, bemerkte Duff.

      [AN] Verr�ter, Angeber.

    �Das wird die Dame ja wohl wissen�, bemerkte Blathers. �Unterbrich
    mich doch nicht immer. Also, Mi�, der Conkey Chickweed hatte ein
    Wirtshaus oberhalb Battle-Bridge und 'nen Raum, den viele junge
    Lords besuchten, um den Hahnenk�mpfen, Dachshetzen und dergleichen
    zuzuschauen, was man nirgends besser sehen konnte. Er geh�rte zu der
    Zeit noch nicht zur Kabrusche[AO], und einst wurden ihm mitten in der
    Nacht dreihundertsiebenundzwanzig Guineen aus seiner Schlafkammer von
    'nem gro�en Manne mit 'nem schwarzen Pflaster �ber dem einen Auge
    gestohlen, der sich unter dem Bett versteckt gehabt hatte und mit dem
    Gelde aus dem Fenster sprang. Er war dabei flink genug; Conkey aber
    war auch geschwind; das Ger�usch hatte ihn aufgeweckt; er sprang aus
    dem Bette, scho� hinter dem Diebe drein und machte die Nachbarn wach.
    Sie erhoben sogleich ein allgemeines Hallo und fanden, da� Conkey den
    Dieb getroffen haben mu�te, denn sie entdeckten und verfolgten auf
    einer ganzen Strecke Blutspuren, die sich indes endlich verloren. Das
    Geld war fort, und Chickweed machte Bankerott. Er ging ein paar Tage
    ganz au�er sich umher, zerraufte sich das Haar und erregte so sehr das
    allgemeine Mitleid, da� ihm von allen Seiten milde Gaben zugeschickt,
    Subskriptionen f�r ihn er�ffnet wurden usw. Eines Tages kam er in das
    Polizeibureau hereingest�rzt und hatte eine geheime Unterredung mit dem
    Friedensrichter, der darauf Jem Spyers (Jem war einer der t�tigsten
    Geheimpolizisten) beorderte, Chickweed bei Gefangennehmung des Diebes
    Beistand zu leisten. >Spyers<, sagte Chickweed, >ich habe ihn gestern
    morgen vor meinem Hause vorbeigehen sehen.< -- >Warum haben Sie ihn
    nicht sogleich angehalten?< fragte Spyers. >Ich war so best�rzt, da�
    Sie mir den Hirnsch�del mit 'nem Zahnstocher h�tten entzweischlagen
    k�nnen,< antwortete der arme Mensch; >wir werden ihn aber gewi�
    attrapieren, denn heut' abend zwischen zehn und elf Uhr kommt er
    wieder vor�ber.< Spyers ging sogleich mit ihm und pflanzte sich an
    ein Wirtshausfenster hinter den Vorhang. Er rauchte in guter Ruh',
    aber mit dem Hut auf dem Kopfe, seine Pfeife, als Chickweed pl�tzlich
    anfing zu schreien: >Da ist er! Haltet den Dieb! Mordjo, mordjo!< Jem
    Spyers st�rzte hinaus und sah Chickweed im vollen Laufe hinter dem
    Diebe herrennen. Er fing auch an zu laufen, was hast du, was kannst du,
    geriet endlich ins Gedr�nge und fand Chickweed darin wieder, allein der
    Dieb war entkommen, was merkw�rdig genug war. Am anderen Morgen war
    Spyers abermals auf seinem Posten, sah sich die Augen nach 'nem gro�en
    Manne mit 'nem schwarzen Pflaster m�de, so da� er sie endlich mal
    wegwenden und ruhen lassen mu�te, und im selbigen Augenblick, als er's
    tat, fing Chickweed wiederum an zu schreien. Jem st�rzt hinaus und ihm
    nach, sie laufen zweimal so weit wie am vorigen Tage, und endlich ist
    der Dieb wiederum zum Geier. Und so ging's noch mehrere Male, so da�
    die Nachbarn sagten, der Teufel selbst h�tte Chickweed bestohlen und
    spielte ihm hinterher noch schlechte Streiche; andere aber sagten, der
    ungl�ckliche Chickweed w�re vor Kummer verr�ckt geworden.�

      [AO] Gaunergenossenschaft.

    �Was sagte denn Jem Spyers?� fragte der Doktor, der wieder in das
    Zimmer zur�ckgekehrt war.

    �Jem Spyers,� erwiderte der Erz�hler, �sagte 'ne lange Zeit gar nichts
    und horchte auf alles, ohne da� man's ihm ansah, zum Zeichen, da�
    er sich auf sein Gesch�ft verstand. Eines Morgens aber trat er zu
    Chickweed und sagte: >Guter Freund, ich hab's jetzt heraus, wer den
    Diebstahl begangen hat.< -- >Wirklich!< rief Chickweed aus; >o mein
    bester Spyers, machen Sie nur, da� ich mich an dem Halunken r�chen
    kann, so werd' ich dermaleinst zufrieden sterben. Bester Spyers, wie
    hei�t der B�sewicht?< -- >Guter Freund,< antwortete Spyers, ihm eine
    Prise anbietend, >lassen Sie die Narretei! Sie haben es selbst getan.<
    Und so war's auch, Chickweed hatte sich dadurch ein anst�ndiges St�ck
    Geld gemacht, und es w�rde auch niemand dahintergekommen sein, wenn er
    nicht so �bereifrig gewesen w�re, den Verdacht von sich fernzuhalten.�

    �Ein seltsamer Fall�, bemerkte der Doktor. �Wenn es Ihnen aber beliebt,
    so k�nnen Sie jetzt hinaufgehen.�

    Die beiden Konstabler begaben sich mit Losberne in Olivers Zimmer.
    Giles leuchtete ihnen. Der kleine Patient hatte geschlummert und sah
    kr�nker und fieberischer aus als am Tage. Der Doktor st�tzte ihn, so
    da� er sich eine kurze Weile emporrichten konnte, und er starrte umher,
    ohne zu wissen, was mit ihm vorging, oder sich zu erinnern, wo er sich
    befand oder was mit ihm vorgegangen war.

    �Dies ist der Knabe,� sagte Losberne leise, aber dessenungeachtet mit
    gro�er Lebhaftigkeit, �der in einem Garten hier in der N�he bei einer
    kleinen �bertretung, wie sie bei Kindern h�ufig vorkommt, durch einen
    Selbstschu� verwundet ist, in Mrs. Maylies Hause Beistand gesucht, und
    den der scharfblickende Herr da mit dem Lichte in der Hand sogleich
    festgehalten und derma�en mi�handelt hat, da� das Leben des Patienten,
    was ich �rztlich bescheinigen kann, betr�chtlich gef�hrdet worden ist.�

    Blathers und Duff hefteten die Blicke auf den solcherma�en ihrer
    Beachtung empfohlenen Giles, dessen Mienen das spa�hafteste Gemisch von
    Furcht und Verwirrung ausdr�ckten.

    �Sie werden nicht leugnen wollen?� f�gte Losberne, Oliver wieder
    niederlegend, hinzu.

    �Es geschah alles zum -- zum Besten, Sir!� antwortete Giles. �Ich hielt
    ihn f�r den Knaben; h�tte mich sonst sicher nicht mit ihm befa�t. Ich
    bin wahrlich kein Unmensch, Sir.�

    �F�r was f�r 'nen Knaben hielten Sie ihn?� fragte Blathers.

    �F�r den Gehilfen der Einbrecher�, erwiderte Giles. �Sie -- sie hatten
    einen Knaben bei sich.�

    �Halten Sie ihn jetzt noch f�r den Knaben?�

    �Kann's wirklich nicht sagen -- k�nnt's nicht beschw�ren, da� er es
    ist.�

    �Was glauben Sie aber?�

    �Ich wei� wirklich nicht, was ich glauben soll. Ich glaube nicht, da�
    es der Knabe ist; ich bin so gut wie gewi�, da� er es nicht ist, Sie
    wissen, da� er es nicht sein kann.�

    �Hat der Mann getrunken, Sir?� fragte Blathers den Doktor.

    Losberne hatte unterdes Olivers Puls gef�hlt, stand auf und bemerkte,
    die Herren m�chten, wenn sie Zweifel hegten, im ansto�enden Zimmer
    Brittles befragen. Man begab sich in das ansto�ende Zimmer, und
    Brittles wurde gerufen und verwickelte sich, wie Mr. Giles, in ein
    solches Irrsal neuer Widerspr�che und Unm�glichkeiten, da� durchaus
    nichts klar wurde als seine eigene Unklarheit, und da� nur einige
    seiner Aussagen einiges Licht gaben: er w�rde den Knaben nicht
    wiedererkennen, h�tte Oliver nur f�r denselben gehalten, weil Giles
    gesagt, da� er es w�re, und Giles h�tte noch vor f�nf Minuten in der
    K�che erkl�rt, da� er zu voreilig gewesen zu sein f�rchte.

    Unter anderen scharfsinnigen Fragen wurde auch die aufgeworfen, ob
    Mr. Giles wirklich jemand getroffen habe, und als sein zweites Pistol
    untersucht wurde, fand sich, da� es nur mit Pulver geladen war, --
    eine Entdeckung, welche gro�en Eindruck auf alle machte, den Doktor
    ausgenommen, der zehn Minuten zuvor die Kugel herausgezogen hatte.
    Den gr��ten Eindruck machte sie aber auf Mr. Giles selbst, der in der
    schrecklichsten Angst geschwebt hatte, ein ungl�ckliches Kind verwundet
    zu haben, und nunmehr nach Kr�ften die Vermutung beg�nstigte, da� auch
    das erste Pistol nur mit Pulver geladen gewesen sei. Endlich entfernten
    sich Blathers und Duff, ohne sich um Oliver viel zu k�mmern, den
    Konstabler aus Chertsey zur�cklassend und unter dem Versprechen, am
    anderen Morgen wiederzukommen.

    Am anderen Morgen verbreitete sich in dem St�dtchen, in welchem sie
    �bernachtet, das Ger�cht, da� zwei M�nner und ein Knabe in der Nacht
    unter verd�chtigen Umst�nden angehalten und nach Kingston gebracht
    w�ren, wohin sich demgem�� Blathers und Duff begaben. Die verd�chtigen
    Umst�nde schrumpften indes bei genauerer Nachforschung zu dem einen
    Umstande zusammen, da� die Delinquenten in einem Heuschober geschlafen
    hatten, was, obwohl ein gro�es Verbrechen, doch nur mit Gef�ngnis
    bestraft werden kann und in den gnadenvollen Augen des englischen, mit
    gemeinsamer Liebe alle Untertanen umfassenden Gesetzes, in Ermangelung
    aller sonstigen Indizien, nicht als gen�gender Beweis gilt, da� der
    oder die Schl�fer gewaltsamen Einbruch begangen haben und deshalb der
    Todesstrafe verfallen sind. Blathers und Duff kehrten daher gerade so
    klug zur�ck, wie sie hingereist waren.

    Kurz, nach mehrfachen Verhandlungen lie� sich der n�chstwohnende
    Friedensrichter leicht bewegen, Mrs. Maylies und Mr. Losbernes
    B�rgschaft f�r Olivers Erscheinen vor Gericht anzunehmen, falls
    er zitiert werden sollte, und Blathers und Duff gingen, nachdem
    sie durch ein paar Guineen belohnt waren, mit geteilten Meinungen
    nach London zur�ck, indem der letztere, nach reiflicher �berlegung
    aller betreffenden Umst�nde, zu der Annahme hinneigte, da� der
    Einbruchsversuch von der Familie Pet ausgegangen sei, wogegen der
    erstere ebenso sehr geneigt war, das ganze Verdienst der Tat dem gro�en
    Conkey Chickweed zuzuschreiben.

    Mit Oliver besserte es sich unter der vereinten sorgf�ltigen Behandlung
    und Pflege Mrs. Maylies, Roses und des gutherzigen Doktors. Wenn
    gl�hende Bitten, aus Herzen von Dankbarkeit �berflie�end, im Himmel
    erh�rt werden -- und was sind Gebete, wenn der Himmel sie nicht erh�rt?
    --, so vernahm er die Segnungen, die das verwaiste Kind auf seine
    Wohlt�ter herabflehte, die dadurch mit Friede und Freude in ihrem
    Innern belohnt wurden.




    32. Kapitel.

        Von dem gl�cklichen Leben, das Oliver bei seinen g�tigen
        G�nnerinnen zu f�hren anfing.


    Oliver litt nicht wenig. Zu den Schmerzen der Schu�wunde kam noch
    ein heftiges Fieber, die Folge der K�lte und N�sse, der er nach
    seiner Verwundung ausgesetzt gewesen war. Er lag mehrere Wochen fest
    zu Bett, fing indes allm�hlich an zu genesen und konnte bald unter
    Tr�nen mit wenigen Worten ausdr�cken, wie tief er die G�te der beiden
    freundlichen, liebevollen Damen empf�nde, und wie sehr er w�nschte
    und hoffte, wenn er wiederhergestellt w�re, imstande zu sein, ihnen
    Beweise seiner Dankbarkeit zu geben, etwas zu tun, und wenn es noch so
    wenig w�re, ihnen die Liebe zu zeigen, die sein Herz erf�llte, ihnen
    die �berzeugung zu verschaffen, da� sie ihre G�te an keinen Unw�rdigen
    verschwendeten, sondern da� der arme, verlassene Knabe, den sie vom
    Elende oder Tode errettet, den gl�henden Wunsch hege, ihnen nach all
    seinen Kr�ften und mit tausend Freuden zu dienen.

    �Armes Kind!� sagte Rose, als er eines Tages mit bleichen Lippen Worte
    des Dankes zu stammeln versuchte. �Du sollst viele Gelegenheiten
    erhalten, uns zu dienen, wenn du willst. Wir gehen auf das Land, und
    meine Tante beabsichtigt, dich mitzunehmen. Die l�ndliche Ruhe, die
    reine Luft und die Freuden und Sch�nheiten des Fr�hlings werden bald
    deine g�nzliche Genesung bewirken, und wir wollen dir hundert kleine
    Gesch�fte auftragen, sobald du der M�he gewachsen bist.�

    �Der M�he!� sagte Oliver. �Ach, wenn ich nur f�r Sie arbeiten -- Ihnen
    nur Freude machen k�nnte, dadurch, da� ich Ihre Blumen beg�sse, Ihre
    V�gel f�tterte, den ganzen Tag hin und wieder f�r Sie liefe, was w�rde
    ich darum geben!�

    �Du sollst gar nichts darum geben,� versetzte Rose l�chelnd, �denn wie
    ich es dir schon gesagt habe, wir denken dich auf die vielfachste Weise
    zu besch�ftigen, und du wirst mir die gr��te Freude bereiten, wenn du
    nur halb so viel tust, wie du jetzt versprichst.�

    �Ihnen Freude bereiten -- o wie g�tig Sie sind!� rief Oliver aus.

    �Du wirst mir mehr Freude bereiten, als ich es dir sagen kann�,
    versetzte die junge Dame. �Es gew�hrt mir schon uns�gliches Vergn�gen,
    zu denken, da� meine liebe, gute Tante ein Werkzeug in den H�nden der
    Vorsehung gewesen ist, einen Knaben aus einer so entsetzlichen Lage
    zu erretten, wie du sie uns beschrieben hast; allein zu erfahren,
    da� ihr kleiner Sch�tzling dankbar und liebevoll gegen sie f�r ihre
    Wohlt�tigkeit und ihr Mitleid ist, wird mich weit gl�cklicher machen,
    als du es dir vorstellen kannst. Verstehst du mich, Oliver?� fragte
    sie, Olivers nachdenkliches Gesicht betrachtend.

    �O ja, ja, ich verstehe Sie wohl; aber ich meinte nur, da� ich jetzt
    undankbar w�re.�

    �Gegen wen denn?�

    �Gegen den g�tigen Herrn und die gute alte Frau, denen ich so gro�e
    Wohltaten verdanke, die sich meiner so liebevoll annahmen. Gewi�, sie
    w�rden sich freuen, wenn sie es w��ten, wie gut ich es hier habe.�

    �Das glaube ich auch, und Mr. Losberne hat schon versprochen, dich
    mitzunehmen zu ihnen, sobald es deine Kr�fte erlauben w�rden.�

    �Hat er das versprochen? Oh, ich wei� nicht, was ich vor Freude tun
    werde, wenn ich sie einmal wiedersehe!�

    Oliver war nach einiger Zeit hinl�nglich wiederhergestellt, um stark
    genug zu einer Ausfahrt zu sein, und fuhr eines Morgens mit Mr.
    Losberne in Mrs. Maylies Wagen ab. Als sie an die Br�cke von Chertsey
    kamen, erbla�te Oliver pl�tzlich und stie� einen lauten Ausruf der
    �berraschung und Best�rzung aus.

    �Was gibt es?� rief der Doktor mit seiner gew�hnlichen Lebhaftigkeit.
    �Siehst du etwas -- h�rst du etwas -- hast du Schmerz -- was gibt's?�

    �Da, Sir!� sagte Oliver, aus dem Wagenfenster zeigend. �Da, jenes Haus!�

    �Was ist mit dem Hause? Halt, Kutscher! -- He -- was willst du sagen?�

    �Die Diebe -- in das Haus schleppten sie mich�, fl�sterte Oliver.

    �Ist es m�glich! Halt, halt, Kutscher!� rief der Doktor, sprang aus dem
    Wagen, noch ehe derselbe hielt, lief nach dem ver�det aussehenden Hause
    und fing an wie toll gegen die T�r zu h�mmern.

    �Zum Teufel, was soll das?� sagte ein kleiner, h��licher, buckliger
    Mann, der die T�r so pl�tzlich �ffnete, da� Losberne fast in das Haus
    hineingefallen w�re.

    �Was das soll?� rief der Doktor, ihn ohne Umst�nde beim Kragen fassend.
    �Sehr viel soll's. Es handelt sich um Diebstahl und Einbruch.�

    �Und es wird sich auch sogleich um Mord handeln,� erwiderte der
    Bucklige kaltbl�tig, �wenn Sie nicht sogleich von mir ablassen. H�ren
    Sie?�

    �Ich h�re sehr wohl�, sagte der Doktor, ihn kr�ftig sch�ttelnd. �Wo ist
    -- wie hei�t der verw�nschte Halunke gleich -- ja, Sikes -- Spitzbube,
    wo ist Sikes?�

    Der Bucklige starrte ihn erstaunt und w�tend an, entschl�pfte ihm,
    stie� eine Flut der schrecklichsten Verw�nschungen aus und ging in
    das Haus zur�ck. Bevor er jedoch die T�r wieder verschlie�en konnte,
    st�rmte der Doktor in das n�chste Zimmer hinein und blickte forschend
    umher, allein von allem, was er sah, wollte nichts mit Olivers
    Beschreibung zusammenstimmen.

    �Was soll das bedeuten, da� Sie auf solche Weise in mein Haus
    eindringen?� fragte nach einigen Augenblicken der Bucklige, der ihn
    scharf beobachtet hatte. �Wollen Sie mich bestehlen oder ermorden?�

    �Hast du jemals einen Mann in solcher Absicht aus einer Equipage
    aussteigen sehen, du l�cherliche, alte Mi�geburt?� lautete des
    reizbaren Doktors Gegenfrage.

    �Was wollen Sie denn aber sonst?� fuhr ihn der Bucklige an. �Packen Sie
    sich augenblicklich aus meinem Hause, oder es wird Sie reuen.�

    �Ich werde gehen, sobald es mir beliebt,� sagte Losberne, in das andere
    Zimmer hineinblickend, das gleichfalls keine �hnlichkeit mit dem
    von Oliver beschriebenen hatte, �und will dir schon noch hinter die
    Schliche kommen!�

    �So!� h�hnte der Kr�ppel. �Wenn Sie mich suchen, ich bin hier zu
    finden. Ich habe hier nicht als ein Verr�ckter und ganz allein seit
    f�nfundzwanzig Jahren gewohnt, um mich von Ihnen hudeln zu lassen. Sie
    sollen mir daf�r b��en, sollen mir daf�r b��en!�

    Der mi�gestaltete kleine D�mon fing darauf an, auf das schrecklichste
    und ungeb�rdigste zu schreien und zu toben, der Doktor murmelte vor
    sich hin: �Dumme Geschichte! Der Knabe mu� sich geirrt haben�, warf
    dem Buckligen ein St�ck Geld zu und kehrte zu dem Wagen zur�ck. Der
    Bucklige folgte ihm unter best�ndigen Schimpfreden und Verw�nschungen,
    sah, w�hrend Losberne dem Kutscher ein paar Worte sagte, in den Wagen
    hinein und warf Oliver einen so grimmigen, stechenden, rachs�chtigen
    und giftigen Blick zu, da� ihn der kleine Rekonvaleszent monatelang
    wachend oder schlafend nicht wieder vergessen konnte. Losberne stieg
    ein, und sie fuhren ab, h�rten aber den Buckligen noch lange schreien
    und toben, der sich vor Wut sch�umend das Haar zerraufte, mit den F��en
    stampfte und ganz au�er sich zu sein schien.

    �Ich bin ein Esel!� sagte der Doktor nach einem langen Stillschweigen.
    �Hast du das schon gewu�t, Oliver?�

    �Nein, Sir.�

    �Dann vergi� es ein anderes Mal nicht. -- Selbst wenn es das Haus
    war,� fuhr er nach einer abermaligen Pause fort, �und die Diebe darin
    gewesen w�ren -- was h�tt' ich als einzelner tun k�nnen? Und h�tt'
    ich Beistand gehabt, so w�re auch nichts weiter dabei herausgekommen,
    als da� meine Voreiligkeit und die Weise kund geworden, wie ich den
    unangenehmen Handel zu vertuschen gesucht. Es w�re mir freilich gerade
    recht geschehen, und ich w�rde nicht d�mmer danach geworden sein, denn
    ich bringe mich selbst in eine Patsche nach der andern, indem ich immer
    blo� nach den fatalen Eindr�cken des Augenblicks handle.�

    Der treffliche Doktor hatte in seinem ganzen Leben nur nach ihnen
    gehandelt, und es lag kein geringes Lob der in ihm vorherrschenden oder
    ihn bestimmenden Eindr�cke in dem Umstande, da� er, weit entfernt,
    jemals in ernstliche Unannehmlichkeiten durch sie geraten zu sein, bei
    allen, die ihn kannten, die w�rmste und gr��te Hochachtung geno�.
    Mu� die Wahrheit gesagt sein, so war er ein paar Minuten �bler Laune,
    sich in der Hoffnung get�uscht zu sehen, sogleich bei der ersten sich
    darbietenden Gelegenheit Zeugnisse f�r die Wahrheit der Erz�hlung
    Olivers zu erhalten. Sein Gleichmut war jedoch bald wiederhergestellt,
    und da die Antworten des Knaben auf seine erneuerten Fragen klar
    und zusammenh�ngend waren und blieben und mit derselben offenbaren
    Aufrichtigkeit wie fr�her gegeben wurden, so nahm er sich vor, ihnen
    von nun an vollkommenen Glauben zu schenken.

    Da Oliver die Stra�e zu nennen wu�te, in welcher Mr. Brownlow wohnte,
    so waren keine Kreuz- und Querfragen erforderlich, und als sie
    hineinfuhren, klopfte des Knaben Herz so heftig, da� er kaum zu atmen
    imstande war. Losberne forderte ihn auf, das Haus zu bezeichnen.

    �Das da!� rief Oliver, eifrig aus dem Fenster zeigend. �Das wei�e Haus!
    Oh, lassen Sie rasch fahren, recht rasch. Es ist mir, als wenn ich
    sterben m��te, eh' ich hinkomme; ich kann mir vor Zittern nicht helfen!�

    �Nur Geduld, mein lieber Kleiner�, sagte Losberne, ihn auf die Schulter
    klopfend. �Du wirst deine Freunde sogleich sehen, und sie werden sich
    freuen, dich gesund und wohlbehalten wiederzufinden.�

    �Oh, das hoff' ich auch�, versetzte Oliver. �Sie waren so g�tig, so
    sehr, sehr g�tig gegen mich, Sir.�

    Der Wagen hielt, und Oliver blickte mit Tr�nen der freudigsten
    Erwartung nach den Fenstern hinauf. Doch ach! Das wei�e Haus war
    unbewohnt; ein Anschlag verk�ndigte, da� es zu vermieten sei. Losberne
    stieg aus, zog den Knaben mit sich fort, klopfte an die n�chste T�r
    und fragte die �ffnende Magd, ob sie wisse, wohin sich Mr. Brownlow
    gewendet, der nebenan gewohnt habe. Sie wu�te es nicht, lief hinauf, um
    sich zu erkundigen, kehrte zur�ck und brachte die Nachricht, er habe
    sein Haus und seine Mobilien verkauft und sei vor sechs Wochen nach
    Westindien gegangen. Oliver schlug die H�nde zusammen und w�re bald zu
    Boden gesunken.

    �Hat er auch seine Haush�lterin mitgenommen?� fragte Losberne nach
    einem kurzen Stillschweigen.

    �Ja, Sir; der alte Herr, die Haush�lterin und ein Freund von ihm sind
    miteinander abgereist.�

    �Wir kehren sogleich nach Hause zur�ck�, rief der Doktor dem Kutscher
    zu; �und fahren Sie rasch, da� wir sobald wie m�glich aus dem
    verw�nschten London wieder hinauskommen.�

    �Der Buchh�ndler, Sir -- wollen wir nicht zu ihm?� fiel Oliver ein.
    �Ich wei�, wo er wohnt. O bitte, lassen Sie uns zu ihm fahren.�

    �Mein liebes Kind, wir haben f�r einen Tag der T�uschung genug gehabt�,
    erwiderte Losberne. �F�hren wir zum Buchh�ndler, so w�rden wir sicher
    h�ren, da� er sein Haus angez�ndet hat, oder davongegangen oder
    tot w�re. Nein, wir wollen f�r heute sogleich wieder nach Chertsey
    zur�ckkehren.�

    Er wiederholte, gem�� dem Eindruck des Augenblicks, seinen Befehl, und
    sie kehrten nach Chertsey zur�ck.

    Die erfahrene bittere T�uschung verursachte Oliver mitten in seinem
    Gl�cke viel Kummer; denn wie oft hatte er sich w�hrend seiner Krankheit
    an der Vorstellung gelabt, was Mr. Brownlow und Mrs. Bedwin zu ihm
    sagen w�rden, und welche Wonne es sein m��te, ihnen zu erz�hlen, wie
    viele lange Tage und Abende er zugebracht in der Erinnerung an das, was
    sie f�r ihn getan, und in Tr�nen �ber seine schreckliche Entf�hrung
    aus ihrem Hause. Die Hoffnung, sich von Verdacht bei ihnen reinigen zu
    k�nnen, hatte ihn in mancher b�sen Stunde aufrecht erhalten; und nun
    war der Gedanke, da� sie au�er Landes gegangen in dem Glauben, da� er
    ein Dieb und Betr�ger sei -- einem Glauben, in welchem sie vielleicht
    bis zu ihrer Sterbestunde verharrten -- fast mehr, als er zu ertragen
    vermochte.

    Das Benehmen seiner Wohlt�ter und G�nner gegen ihn blieb jedoch
    unver�ndert. Als nach vierzehn Tagen sch�nes Fr�hlingswetter war, die
    B�ume im jungen, frischen Gr�n zu prangen und die Blumen zu bl�hen
    anfingen, trafen sie die erforderlichen Vorbereitungen, ihre Wohnung
    in Chertsey auf einige Monate zu verlassen. Das Silberger�t, das die
    Begierde des Juden erregt hatte, wurde in sicheren Gewahrsam gebracht,
    Giles mit einem zweiten Diener zur Bewachung des Hauses zur�ckgelassen,
    und sie reisten ab auf das Land und nahmen Oliver mit.

    Wer verm�chte das selige Entz�cken, den Seelenfrieden und die s��e,
    trauliche Ruhe zu schildern, die der noch immer schwache Knabe in der
    balsamischen Luft, auf den gr�nen H�geln und in den sch�nen Waldungen
    empfand, die das kleine Dorf, seinen neuen Wohnsitz, umgaben! Wer
    k�nnte es mit Worten beschreiben, welche Stille, welche Frische, welche
    Lust ein Fr�hling auf dem Lande in die Herzen geplagter Stadtbewohner
    senkt! Selbst von Leuten, die in engen, menschengef�llten Stra�en
    ihr Leben unter stetem Ger�usch und in fortw�hrender Plackerei
    zugebracht haben, und in denen nie ein Wunsch nach Ver�nderung ihrer
    Lage aufgestiegen ist, und die das Mauerwerk und die Steine, die
    engen Grenzmarken ihrer kleinen, t�glichen Ausfl�ge, fast zu lieben
    angefangen -- selbst von ihnen, wenn die Todesstunde sich ihnen nahte,
    wei� man es, da� sie sich endlich nach einem fl�chtigen Blicke des
    Antlitzes der Natur sehnten, da� sie, hinweggef�hrt von dem Schauplatze
    ihrer M�hen, Schmerzen und Freuden, gleichsam verj�ngt zu werden
    schienen, Tag f�r Tag ein gr�nes, sonniges Pl�tzchen aufsuchten und
    in dem blo�en Schauen des blauen Himmels, der blumen�bers�ten Wiesen
    und des glitzernden Stromes einen Vorgeschmack des Himmels selbst
    empfanden, der ihre letzten Leiden vers��te, so da� sie friedlich wie
    die untergehende Sonne in ihre Gr�ber sanken, gleich der Sonne, die sie
    mit Entz�cken am Fenster ihres einsamen, stillen K�mmerchens sinken
    sahen. Die Erinnerungen, welche durch friedliche l�ndliche Szenen
    hervorgerufen werden, sind nicht von dieser Welt und ihren Gedanken
    oder Hoffnungen. Ihr s��es, lindes Einwirken kann uns lehren, frische
    Kr�nze f�r die Gr�ber unserer Lieben zu winden, unsere Herzen zu
    l�utern und unseren alten Ha�, unsere Feindschaften zu verscheuchen und
    auszutilgen; und durch das alles zieht sich auch bei minder sinnigen
    Gem�tern ein halbes, unbestimmtes Bewu�tsein, Gef�hle solcher Art einst
    in einer fernen, l�ngstentflohenen Zeit empfunden zu haben -- ein
    Bewu�tsein, das feierlich-ernste Ahnungen einer entfernten kommenden
    Zeit erweckt, und Stolz und Weltsinn d�mpft und unterdr�ckt.

    Das D�rfchen, wohin sie sich begaben, lag �u�erst angenehm, und Oliver
    war es, als wenn ein neues Leben f�r ihn begonnen h�tte, denn er
    hatte seine Tage von fr�hester Kindheit an in engen, oft schmutzigen
    R�umen und unter Ger�usch und L�rm zugebracht. Rosen und Gei�blatt
    bedeckten die W�nde des H�uschens seiner G�nnerin, die St�mme der B�ume
    waren mit Efeu bewachsen, und die Gartenblumen erf�llten die Luft mit
    k�stlichen D�ften. Dicht neben dem H�uschen lag ein kleiner Friedhof,
    nicht angef�llt mit hohen, widerw�rtigen Grabsteinen, sondern voll von
    bescheidenen Gras- und Moosh�gelchen, unter welchen die alten Leute des
    Dorfes von ihren M�hen ausruhten. Oliver besuchte ihn oft und setzte
    sich, des elenden Grabes seiner Mutter gedenkend, bisweilen nieder und
    weinte ungesehen; doch wenn er dann die Augen emporhob zu dem klaren
    blauen Himmel �ber ihm, so dachte er sie sich nicht mehr ruhend im
    Scho�e der Erde, sondern droben in den Wohnungen der Seligen und weinte
    wohl fort um sie, doch ohne Schmerz.

    Es war eine sch�ne, gl�ckliche Zeit. Die Tage vergingen friedlich
    und heiter, und die Abende brachten weder Furcht noch Sorge, kein
    Schmachten in einem d�steren Kerker, nicht den Anblick heimkehrender,
    verworfener Menschen mit sich, sondern nur s��e, traute Gedanken. Jeden
    Morgen ging Oliver zu einem silberhaarigen, alten Manne, der dicht
    neben der kleinen Kirche wohnte und ihn lesen und schreiben lehrte,
    und so freundlich mit ihm redete und sich so sehr um ihn bem�hte, da�
    Oliver sich selbst nie genug tun konnte, ihm Freude zu machen. Zu
    anderen Tagesstunden lustwandelte er mit Mrs. Maylie und Mi� Rose und
    h�rte ihrer Unterhaltung zu oder sa� bei ihnen an einem schattigen
    Pl�tzchen und horchte dem Vorlesen der j�ngeren Dame, ohne sich jemals
    satth�ren zu k�nnen. Zu anderen Zeiten war er eifrig mit seiner
    Lektion auf den folgenden Tag in einem kleinen Zimmer besch�ftigt,
    dessen Fenster in den Garten ging; und wenn der Abend herankam, ging
    er wieder mit den Damen aus und war �bergl�cklich, wenn er ihnen eine
    Blume pfl�cken konnte, nach welcher sie etwa Begehrung trugen, oder
    wenn sie etwas vergessen hatten und ihm auftrugen, es zu holen. War es
    d�mmerig geworden, so pflegte sich Rose an das Fortepiano zu setzen
    und zu spielen oder ein altes Lied zu singen, das ihre Tante zu h�ren
    w�nschte, und Oliver sa� dann am Fenster und horchte den lieblichen
    T�nen, und Z�hren wehm�tiger Lust rannen �ber seine Wangen hinab.

    Wie ganz anders wurde der Sonntag hingebracht, als ihn Oliver je
    verlebt hatte, und welch ein sch�ner Tag war er gleich den anderen
    Tagen in dieser gl�cklichen Zeit! Morgens wurde die kleine Kirche
    besucht, vor deren Fenstern sich gr�ne Bl�tter im Winde bewegten,
    und drau�en zwitscherten die V�gel, und durch die niedrige T�r drang
    die reine, erquickende Luft herein. Die armen Landleute erschienen
    so sauber und reinlich und knieten bei den Gebeten so ehrfurchtsvoll
    nieder, da� ihr Gottesdienst wie eine Freude und nicht wie eine
    beschwerliche Pflicht�bung erschien; und wenn der Gesang auch weniger
    als kunstlos war, so kam er doch vom Herzen und klang zum wenigsten
    Olivers Ohre wohlt�nender als alle Kirchenmusik, die er in seinem
    ganzen Leben geh�rt hatte. Und dann wurden die Spazierg�nge wie
    gew�hnlich gemacht, und manche reinliche H�tte im Dorfe ward besucht;
    abends las dann Oliver einige Kapitel aus der Bibel vor, die ihm in der
    Woche vorher erkl�rt waren, und er empfand dabei eine so stolze Freude,
    als wenn er der Geistliche selbst gewesen w�re.

    Morgens fr�h um sechs Uhr war er auf und drau�en und streifte in den
    Feldern umher, Str�u�e von wilden Blumen pfl�ckend, mit denen er den
    Fr�hst�ckstisch schm�ckte. Auch brachte er frisches Kreuzkraut f�r
    Roses V�gel mit nach Hause, und waren dieselben besorgt, so hatte er
    fast t�glich einen kleinen Mildt�tigkeitsauftrag im Dorfe auszurichten,
    oder es war etwas im Garten zu tun, wobei er unter der Anleitung des
    G�rtners den lebhaftesten Eifer bewies, bis Mi� Rose erschien und ihn
    durch manches L�cheln, manchen freundlichen Lobspruch belohnte.

    So vergingen drei Monate -- drei Monate, die im Leben der Gl�cklichsten
    sch�n zu nennen gewesen sein w�rden, f�r Oliver aber, nach seinen
    unruhigen, tr�ben Tagen, die ungemischteste Seligkeit waren. Bei
    reinster und edelster Liebe und Gro�mut auf der einen und bei der
    wahrhaft innigsten und w�rmsten Dankbarkeit auf der anderen Seite
    war es in der Tat kein Wunder, da� Oliver am Schlusse dieses kurzen
    Zeitabschnittes bei der alten Dame und ihrer Nichte vollkommen heimisch
    geworden war, und da� beide durch ihren Stolz auf ihn und ihre Freude
    an ihm die hei�e Zuneigung seines jungen und lebhaft empf�nglichen
    Herzens vergalten.




    33. Kapitel.

        In dem Olivers und seiner G�nnerinnen Gl�ck eine pl�tzliche St�rung
        erleidet.


    Der Fr�hling schwand rasch dahin, und der Sommer kam, und war alles
    umher sch�n gewesen im Lenz, so bl�hte und gl�nzte es nun in vollster,
    �ppigster Pracht. Die B�ume streckten ihre Arme �ber den durstigen
    Boden aus, verwandelten offene und nackte Stellen in dunkle, heimliche
    Pl�tzchen, und wie k�stlich lie�en sich aus ihrem stillen, hehren
    Schatten die sonnigen Felder beschauen! Die Erde hatte sich mit ihrem
    glanzvoll gr�nsten Mantel geschm�ckt, und Millionen Bl�ten durchdufteten
    die Luft. Alles gr�nte, bl�hte, strahlte von Lust und verk�ndete Freude.

    Das ruhige Leben in Mrs. Maylies Landh�uschen nahm seinen Fortgang,
    und heiter und froh genossen die Bewohner die sch�ne Zeit. Oliver
    war gesund und kr�ftig geworden, ohne da� -- wie es sonst wohl der
    Fall ist -- eine �nderung in seinen Gef�hlen oder seinem Benehmen
    eingetreten w�re. Er war fortw�hrend derselbe sanfte, z�rtliche,
    liebevolle Knabe, der er gewesen, als unter Krankheit und Schmerz seine
    Kr�fte geschwunden waren und seine Schw�che ihn auch bei den kleinsten
    W�nschen und Bed�rfnissen von seinen Pflegerinnen abh�ngig gemacht
    hatte.

    Einst an einem sch�nen Abende machte er mit Mrs. Maylie und Rose
    einen ungew�hnlich langen Spaziergang; es war sehr hei� gewesen, doch
    k�hlte jetzt ein linder Wind die Luft, und am Himmel gl�nzte schon
    der Vollmond. Rose war sehr munter und wohlgemut, sie gingen unter
    fr�hlichem Gespr�che weiter, als sie gew�hnlich zu tun pflegten, Mrs.
    Maylie empfand endlich Erm�dung, und sie kehrten langsamer nach Hause
    zur�ck. Rose legte nur ihren Hut ab, setzte sich wie gew�hnlich an das
    Piano, schlug einige Akkorde an, ging zu einer langsam-feierlichen
    Weise �ber und fing, w�hrend sie dieselbe spielte, zu schluchzen an.

    �Was weinst du, liebes Kind?� fragte Mrs. Maylie; allein Rose
    antwortete nicht und spielte nur ein wenig rascher, als wenn sie aus
    einem schmerzlichen Sinnen aufgeweckt worden w�re.

    �Liebes Kind, was ist dir?� fragte Mrs. Maylie, hastig aufstehend und
    sich �ber sie beugend. �Dein Gesicht ist in Tr�nen gebadet. Was betr�bt
    dich denn, bestes Kind?�

    �Nichts, Tante, nichts�, erwiderte Rose. �Ich wei� selbst nicht, wie
    mir ist -- ich kann es nicht beschreiben -- ich f�hle mich so matt, so
    --�

    �Du bist doch nicht krank, Rose?� fiel Mrs. Maylie ein.

    �O nein, nein�, sagte die junge Dame schaudernd, als wenn sie pl�tzlich
    von einem Fieberfroste gesch�ttelt w�rde; �mir wird wenigstens sogleich
    wieder besser sein. Schlie�e das Fenster, Oliver.�

    Oliver eilte, ihr Gehei� zu erf�llen, sie zwang sich, heiter zu
    scheinen und spielte eine muntere Weise; allein die H�nde fielen
    ihr kraftlos in den Scho�, sie stand auf, sank auf das Sofa nieder,
    bedeckte ihr Antlitz und lie� den Tr�nen freien Lauf, die sie nicht
    mehr zu unterdr�cken vermochte.

    �Mein liebes Kind!� rief Mrs. Maylie, sie an die Brust dr�ckend, aus;
    �ich habe dich ja noch nie so gesehen!�

    �Ich beunruhige Sie nur sehr ungern,� erwiderte Rose, �kann aber trotz
    aller M�he dies Weinen nicht unterdr�cken. Ich f�rchte, da� ich doch
    krank bin, Tante.�

    Sie war es in der Tat; denn als Licht gebracht wurde, gewahrten alle,
    da� sich ihre Farbe in der kurzen Zeit seit der R�ckkehr von dem
    Spaziergange in Marmorbl�sse verwandelt hatte. Ihr Antlitz hatte nichts
    von seiner Sch�ne verloren, und doch war mit ihren Z�gen eine Wandlung
    vorgegangen, und es lag ein Ausdruck der Unruhe und Abspannung darin,
    den sie noch niemals gezeigt hatten. Nach Verlauf einer Minute waren
    ihre Wangen wieder von Purpurr�te �bergossen, ihre sanften, blauen
    Augen bekamen einen stechenden, unheimlichen Blick, und auch dieser
    verschwand bald wieder, gleich einem vor�berziehenden W�lkchen, und die
    Leichenbl�sse kehrte zur�ck.

    Oliver, der die alte Dame genau beobachtet hatte, bemerkte, da� sie
    gro�e Unruhe empfand, wie es in Wahrheit bei ihm selber der Fall war;
    da sie sich indes offenbar den Anschein zu geben suchte, als wenn sie
    die Sache leicht n�hme, so tat er dasselbe, was bei Rose eine g�nstige
    Wirkung hervorzubringen schien. Denn als sie auf Zureden ihrer Tante zu
    Bett ging, sah sie wieder wohler aus, versicherte, es auch zu sein und
    f�gte hinzu, sie w�re �berzeugt, da� sie am anderen Morgen gesund und
    munter wie sonst erwachen w�rde.

    �Ich hoffe, Ma'am,� sagte Oliver, als Mrs. Maylie zur�ckkehrte, �da�
    Mi� Rose nicht ernstlich krank werden wird. Sie sah heute abend unwohl
    genug aus; doch --�

    Die alte Dame winkte ihm, nicht fortzufahren, setzte sich, st�tzte
    schweigend den Kopf auf die Hand und sagte endlich mit bebender Stimme:
    �Ich will es auch hoffen, Oliver. Ich habe einige Jahre sehr gl�cklich
    -- vielleicht zu gl�cklich mit ihr verlebt, und es k�nnte Zeit sein,
    da� mir wieder ein Ungl�ck begegnet -- ich hoffe indes, nicht dieses.�

    �Was f�r ein Ungl�ck, Ma'am?� fragte Oliver.

    �Ich meine den schweren Schlag,� antwortete die alte Dame fast tonlos,
    �das liebe M�dchen zu verlieren, das so lange schon meine Freude und
    mein Trost gewesen ist.�

    �Das verh�te Gott!� rief Oliver hastig aus.

    �Ich sage Ja und Amen dazu, mein Kind!� fiel die alte Dame, die H�nde
    ringend, ein.

    �Sie brauchen sicher so etwas Schreckliches nicht zu f�rchten�, fuhr
    Oliver fort. �Mi� Rose war ja vor zwei Stunden vollkommen wohl.�

    �Und jetzt ist sie sehr unwohl�, versetzte Mrs. Maylie, �und wird ohne
    Zweifel noch kr�nker werden. O meine liebe, liebe Rose! Was sollte ich
    anfangen ohne sie!� Sie wurde so sehr und so schmerzlich bewegt, da�
    Oliver, seine eigene Herzensangst unterdr�ckend, sich bem�hte, sie
    zu beruhigen, und sie dringend bat, um der lieben jungen Dame selbst
    willen gefa�ter zu sein.

    �Bedenken Sie doch nur, Ma'am,� sagte er, gewaltsam die Tr�nen
    zur�ckdr�ngend, die ihm in die Augen schossen, �wie jung und wie gut
    sie ist und wie sie alles um sich her erfreut. Ich wei� es -- wei�
    es ganz gewi�, da� sie um ihrer selbst und um Ihret- und unser aller
    willen, die sie so froh und gl�cklich macht, nicht sterben wird; nein,
    nein, Gott l��t sie nimmermehr schon jetzt sterben!�

    �Ach! du sprichst und denkst wie ein Kind, mein guter Oliver,� sagte
    Mrs. Maylie, ihm die Hand auf den Kopf legend, �und irrst, so nat�rlich
    es sein mag, was du sagst. Indes hast du mich an meine Pflicht
    erinnert. Ich hatte sie auf einen Augenblick ganz vergessen, Oliver,
    und hoffe Verzeihung zu finden, denn ich bin alt und habe genug gesehen
    von Krankheiten und vom Tode, um den Schmerz zu kennen, den sie den
    Hinterbleibenden zuf�gen. Auch besitze ich genug Erfahrung, um zu
    wissen, da� es nicht immer die J�ngsten und Besten sind, die den sie
    Liebenden erhalten werden -- was uns jedoch eher tr�sten als bek�mmern
    sollte, denn der Himmel ist weise und g�tig, und Erfahrungen solcher
    Art lehren uns eindringlich, da� es eine noch sch�nere Welt gibt als
    diese und da� wir bald hin�bergehen zu ihr. Gottes Wille geschehe! Doch
    liebe ich sie, und er allein wei� es, wie sehr, wie sehr!�

    Oliver war verwundert, da� Mrs. Maylie, sobald sie diese Worte
    gesprochen, ihren Klagen pl�tzlich Einhalt tat, sich hoch emporrichtete
    und vollkommen ruhig und gefa�t erschien. Er war noch mehr erstaunt,
    als er bemerkte, da� sie sich in ihrer Festigkeit gleich, bei allem
    Sorgen und Wachen besonnen und gesammelt blieb und jede ihrer Pflichten
    dem Anscheine nach sogar mit Heiterkeit erf�llte. Doch er war jung und
    wu�te noch nicht, welch gro�en Tuns und Duldens starke Seelen unter
    schwierigen und entmutigenden Umst�nden f�hig sind; und wie h�tte er es
    wissen sollen, da sich die Starken selbst ihrer Kraft nur selten bewu�t
    sind?

    Es folgte eine angstvolle Nacht, und als der Morgen kam, waren Mrs.
    Maylies Vorhersagungen nur zu wahr geworden. Rose lag im ersten Stadium
    eines heftigen und gefahrdrohenden Fiebers.

    �Wir m�ssen t�tig sein, Oliver, und d�rfen uns nicht einem nutzlosen
    Schmerze �berlassen�, sagte Mrs. Maylie, den Finger auf den Mund legend
    und ihm fest in das Gesicht blickend. �Dieses Schreiben mu� so eilig
    wie irgend m�glich Mr. Losberne zugeschickt werden. Du sollst es nach
    dem Flecken tragen, der auf dem Fu�wege nur vier Meilen entfernt ist;
    von dort soll ein reitender, expresser Bote nach Chertsey abgehen.
    Der Gastwirt besorgt ihn, und ich wei�, da� du den Auftrag p�nktlich
    ausrichten wirst.�

    Oliver konnte nicht antworten, allein seine Mienen verk�ndigten, da� er
    vor Begierde brannte, sich sogleich auf den Weg zu begeben.

    �Hier ist noch ein Schreiben,� fuhr Mrs. Maylie nachsinnend
    fort, �allein ich wei� kaum, ob ich es sogleich abschicken oder
    abwarten soll, wie es mit Roses Befinden wird. Ich m�chte es lieber
    zur�ckhalten, bis ich das Schlimmste f�rchten m��te.�

    �Soll er auch nach Chertsey, Ma'am?� fragte Oliver ungeduldig,
    seinen Auftrag auszurichten, und die zitternde Hand nach dem Briefe
    ausstreckend.

    �Nein!� erwiderte die alte Dame.

    Sie gab ihn jedoch dem Knaben, da sie in Gedanken verloren war, und
    Oliver sah, da� er an Harry Maylie Esq. und nach dem Landsitze eines
    Lords, dessen Namen er noch nie geh�rt hatte, adressiert war.

    �Soll er fort, Ma'am?� fragte Oliver ungeduldig.

    �Nein; ich will bis morgen warten�, sagte die alte Dame, lie� sich das
    Schreiben zur�ckgeben, reichte Oliver ihre B�rse, und er eilte hinaus,
    um in k�rzester Frist nach dem Marktflecken zu gelangen, in welchem
    er staubbedeckt ankam. Er hatte bald das Gasthaus zum Georg gefunden
    und wandte sich an einen Postillon, der ihn an den Hausknecht verwies,
    von welchem er wiederum an den Wirt verwiesen wurde, der bed�chtig zu
    lesen und dann zu schreiben und Befehle zu erteilen anfing, wor�ber
    manche Minute verging. Oliver h�tte selbst auf das Pferd springen und
    davongaloppieren m�gen; doch endlich sprengte ein Berittener des Wirts
    die Stra�e hinunter und war nach wenigen Augenblicken verschwunden.
    Oliver, der vor der T�r gestanden hatte, ging mit leichterem Herzen
    �ber den Hof des Gasthauses, um eiligst heimzukehren. Als er um die
    Ecke eines Stallgeb�udes bog, rannte er gegen einen gro�en, in einen
    Mantel eingeh�llten Mann an, der eben aus der T�r des Gasthauses
    getreten sein mu�te.

    �Ha! zum Teufel, was ist das?� rief der Mann zur�ckprallend und die
    Blicke auf Oliver heftend.

    �Ich bitte um Vergebung, Sir�, sagte Oliver; �ich hatte gro�e Eile und
    sah Sie nicht kommen.�

    �Alle Teufel!� murmelte der Mann vor sich hin, den Knaben mit seinen
    gro�en, schwarzen Augen anstarrend. �Wer h�tte das denken k�nnen? Und
    wenn man ihn zu Staub zerriebe, er w�rde aus 'nem marmornen Sarge
    wieder aufstehen und mir in den Weg treten.�

    �Es tut mir leid, Sir�, stotterte Oliver verwirrt; �ich hoffe, da� ich
    Ihnen keinen Schaden getan habe.�

    �Da� seine Knochen verfaulen!� murmelte der finstere Mann durch
    die verbissenen Z�hne; �h�tte ich nur den Mut gehabt, das Wort
    auszusprechen, so h�tte mich eine einzige Nacht von ihm befreien
    k�nnen. Fluch �ber dein Haupt und die Pest in deinen Leib, du
    H�llenbrand! Was hast du hier zu schaffen?�

    Er hob drohend die Faust empor, knirschte mit den Z�hnen und trat einen
    Schritt vor, als wenn er Oliver einen Schlag versetzen wollte, st�rzte
    aber pl�tzlich zu Boden und wand und kr�mmte sich, w�hrend ihm dicker
    Schaum vor dem Munde stand. Oliver schaute dem Wahnwitzigen (denn ein
    solcher schien ihm der schreckliche Mann zu sein) ein paar Augenblicke
    zu, lief darauf in das Haus, um Beistand zu holen, verlor sodann
    keine Zeit mehr und eilte nach Hause zur�ck, mit gro�er Verwunderung
    und nicht ohne Bangigkeit an das seltsame Benehmen des Unbekannten
    zur�ckdenkend. Er verlor den ganzen Vorfall jedoch bald aus dem
    Ged�chtnis, denn als er in Mrs. Maylies Wohnung wieder angelangt war,
    h�rte und sah er genug, was seinen Gedanken eine ganz andere Richtung
    gab.

    Roses Zustand hatte sich sehr verschlimmert, und noch vor Mitternacht
    lag sie in Fieberphantasien. Der Wundarzt aus dem Dorfe hatte
    Mrs. Maylie erkl�rt, da� die Krankheit ihrer Nichte eine sehr
    beunruhigende Wendung genommen h�tte, und zwar in dem Ma�e, da� ihre
    Wiederherstellung einem Wunder gleichkommen w�rde.

    Wie oft sprang Oliver aus seinem Bett in der Schreckensnacht, um an
    die Treppe zu schleichen und zu horchen, was in dem Krankenzimmer
    vorgehen m�chte! Er bebte fast fortw�hrend an allen Gliedern, und kalte
    Schwei�tropfen traten ihm auf die Stirn, wenn ihm irgendein Ger�usch zu
    verk�nden schien, da� das Schlimmste eingetreten sei. Er hatte nie so
    inbr�nstig zum Himmel gefleht, wie er in dieser Nacht um die Erhaltung
    des teuren Lebens seiner holden, am Rande des Grabes stehenden Freundin
    betete.

    Die Ungewi�heit, die schreckliche, �ngstigende Ungewi�heit, wenn wir
    unt�tig daneben stehen, w�hrend die Wagschale eines Hei�geliebten
    zwischen Tod und Leben schwankt -- die folternden Gedanken, welche dann
    auf das Gem�t einst�rmen, das Herz zu rascheren, heftigen Schl�gen
    treiben, den Atem stocken machen -- die d�steren Bilder, welche sie
    heraufbeschw�ren --, der verzweifelte Herzensdrang, etwas zu tun zur
    Linderung von Schmerzen, die wir nicht lindern k�nnen, zur Entfernung
    einer Gefahr, die wir nicht zu entfernen verm�gen, und die tiefe,
    traurige Niedergeschlagenheit, welche uns dann bei dem Bewu�tsein
    unserer Ohnmacht ergreift: -- welche Qualen lassen sich diesen
    vergleichen, durch welche Erw�gungen oder Anstrengungen k�nnten wir sie
    uns in der Fieberhitze der Aufregung, in unserer tiefen Not erleichtern?

    Der Morgen kam, und das H�uschen war stumm und still. Man fl�sterte
    nur; von Zeit zu Zeit lie�en sich angstvolle Gesichter an der T�r
    blicken, und Frauen und Kinder gingen weinend wieder fort. Den ganzen
    langen Tag und noch stundenlang, nachdem es dunkel geworden war,
    ging Oliver leise im Garten auf und ab, die Augen fortw�hrend hinauf
    nach dem Zimmer der Kranken gewandt und schaudernd beim Anblick des
    verdunkelten Fensters, das ihm aussah, als wenn drinnen der Tod lauernd
    ausgestreckt l�ge. Zu einer sp�ten Abendstunde traf Mr. Losberne ein.
    �'s ist hart�, sagte der weichherzige Doktor, sich abwendend; �'s ist
    hart -- so jung -- so hei� geliebt von so vielen --, doch aber ist nur
    wenig Hoffnung!�

    An einem abermaligen Morgen strahlte die Sonne hell -- so hell und
    heiter, als wenn sie auf kein Leiden, keine Sorge herabblickte; und
    indem die Blumen sie umbl�hten und Leben, Gesundheit und T�ne der
    Freude und lachende Gegenst�nde sie rings umgaben, siechte die junge,
    sch�ne Dulderin dem Grabe entgegen. Oliver schlich hinaus auf den
    stillen Friedhof, setzte sich auf einen der kleinen, gr�nen H�gel und
    weinte um sie in der Stille und Einsamkeit.

    Der Tag war ein so k�stlicher Sommertag, die sonnige Landschaft so
    heiter und gl�nzend, die V�gel sangen und h�pften so munter in den
    Zweigen oder schwangen sich so lebensfroh in die L�fte empor, alles,
    alles schien so laut aufzufordern zur Freude und Lust, da� sich dem
    Knaben, als er die schmerzenden Augen aufschlug, unbewu�t der Gedanke
    aufdr�ngte, dies sei keine Zeit f�r den Tod, und Rose k�nne nimmermehr
    sterben, w�hrend so viele weit geringere Wesen so froh und munter
    w�ren; die Gr�ber w�ren nur f�r den kalten, freudlosen Winter, nicht
    f�r die sonnige, duftige, Lust weckende und gebende Sommerzeit. Fast
    h�tte er geglaubt, die Leichent�cher w�ren f�r die Alten und Abgelebten
    und nicht dazu bestimmt, die jungen und sch�nen Gestalten mit ihrer
    grausigen Nacht zu bedecken.

    Ein Gel�ute der Kirchglocke unterbrach pl�tzlich seine kindlichen
    Gedanken. Es wurde zu den Begr�bnisgebeten gel�utet. Ein l�ndliches
    Leichengefolge schritt durch das Tor herein; die Leidtragenden hatten
    sich mit wei�en Schleifen geschm�ckt; sie begruben einen J�ngling.
    Sie standen mit entbl��ten H�uptern am Grabe, und in ihrer Mitte
    kniete eine weinende Mutter. Aber die Sonne schien hell, und die V�gel
    zwitscherten und h�pften in den Zweigen fort und fort.

    Oliver kehrte nach Hause zur�ck, gedenkend der vielfachen Beweise von
    G�te, die er von der jungen Dame empfangen und mit dem Wunsche, da�
    die Zeit noch einmal kommen m�chte, wo er imstande w�re, ihr ohne
    Aufh�ren zu zeigen, wie dankbar und liebevoll gesinnt er gegen sie war.
    Er hatte sich nichts vorzuwerfen, denn er war eifrig in ihrem Dienste
    gewesen, und doch mu�te er an zehn und wieder zehn F�lle denken, in
    welchen er meinte, nicht eifrig genug gewesen zu sein. Wohl sollten wir
    sorgf�ltig �ber unser Benehmen gegen die, mit denen unsere Lebensbahn
    uns zusammenf�hrt, wachen, und so viel Liebe als m�glich hineinlegen;
    denn jeglichen Todesfall begleitet eine Schar von Gedanken an so viel
    Vers�umtes, so wenig Getanes -- an so viel Vergessenes und an noch viel
    mehr, was h�tte besser getan, oder wieder gut gemacht werden k�nnen,
    da� die Erinnerungen dieser Art zu den allerbittersten geh�ren, die uns
    qu�len k�nnen. Keine Reue ist so schmerzlich, als die vergebliche, und
    wollen wir uns ihre Peinigungen ersparen, so la�t uns beizeiten allen
    dessen gedenken.

    Als Oliver zu Hause angelangt war, fand er Mrs. Maylie im kleinen
    Wohnzimmer. Sein Herz zagte in ihm bei ihrem Anblick, denn sie hatte
    das Bett ihrer Nichte noch keine Minute verlassen, und er zitterte, zu
    denken, welche Veranlassung sie von demselben verscheucht haben k�nnte.
    Er vernahm, da� die Patientin in einen festen Schlummer verfallen
    sei, aus welchem sie erwachen w�rde zur Genesung und zum Leben, oder
    um ihren Lieben das letzte Lebewohl zu sagen und von dieser Welt zu
    scheiden.

    Sie sa�en stundenlang horchend und zu sprechen sich scheuend,
    beieinander. Das Mahl wurde unanger�hrt hinausgetragen, ihre Blicke
    hingen an der Pracht der untergehenden Sonne, doch waren ihre Gedanken
    bei einem anderen Gegenstande. Ihr gespanntes Ohr vernahm den Schall
    herannahender Fu�tritte, und sie eilten zugleich nach der T�r, als
    Losberne eintrat.

    �Was haben Sie von Rose zu melden?� rief ihm die alte Dame entgegen.
    �Sagen Sie es sogleich. Ich kann alles, nur keine Ungewi�heit ertragen.
    In des Himmels Namen, reden Sie! Ist sie tot, ist sie tot?�

    �Nein�, entgegnete der Doktor �u�erst bewegt. �So wahr er g�tig und
    barmherzig ist, wird sie leben, um uns alle noch viele Jahre zu
    begl�cken!�

    Die alte Dame fiel auf die Knie nieder und m�hte sich, die H�nde zu
    falten; allein ihre Kraft, die sie so lange aufrecht erhalten hatte,
    floh mit dem ersten Dankesseufzen, das sie zum Himmel emporsandte, und
    sie sank zur�ck in die Arme des herbeigeeilten Doktors.




    34. Kapitel.

        In welchem ein junger Herr auftritt, und Oliver ein neues Abenteuer
        erlebt.


    Es war fast zu viel Gl�ck, um es ertragen zu k�nnen. Oliver war durch
    die unverhoffte Kunde ganz bet�ubt; er konnte nicht weinen, nicht
    sprechen, nicht bleiben, wo er war. Er mu�te sich erst wieder zu fassen
    suchen, um was er geh�rt, zum klaren Bewu�tsein zu bringen, als er
    sich nach einem langen Umherschweifen in der stillen Abendlandschaft
    durch einen Tr�nenstrom erleichtert, und von der fast nicht mehr zu
    ertragenden Last befreit f�hlte, die ihm gleich einem Alp auf dem
    Herzen gelegen hatte.

    Es dunkelte, als er nach Hause zur�ckkehrte, beladen mit Blumen, die
    er mit ungew�hnlicher Sorgfalt zur Ausschm�ckung des Krankenzimmers
    gepfl�ckt hatte. Als er der Wohnung Mrs. Maylies rasch zuschritt, h�rte
    er hinter sich auf der Stra�e das donnernde Ger�usch eines Wagens.
    Er sah sich um: es war eine Postchaise, und da die Stra�e ziemlich
    schmal war und der Postillon im Galopp fuhr, so trat er dicht an ein
    Gartentor, um nicht in Gefahr zu geraten. Die Chaise n�herte sich, und
    nun erblickte er ein unter einer Nachtm�tze fast verstecktes Gesicht,
    das ihm bekannt schien; er begann nachzusinnen, wem es angeh�ren
    m�chte, als er angerufen wurde, und der Postillon den Befehl zum Halten
    erhielt.

    �Oliver, wie steht es -- wie steht es mit Mi� Rose, Oliver?� rief ihm
    Mr. Giles zu.

    �Ohne Umschweife -- besser oder schlimmer?� rief ein junger Herr, der
    Giles zur�ckzog und sich selbst aus dem Schlage herausbeugte.

    �Besser -- viel besser!� erwiderte Oliver mit freudiger Hast.

    �Gott sei Dank!� rief der junge Herr aus. �Ist's auch gewi�?�

    �Sie k�nnen sich fest darauf verlassen, Sir�, sagte Oliver; �die
    Besserung trat vor ein paar Stunden ein, und Mr. Losberne hat gesagt,
    da� alle Gefahr vor�ber sei.�

    Der junge Mann sagte kein Wort mehr, sondern sprang aus dem Wagen, zog
    Oliver zur Seite und fragte ihn mit bebender Stimme: �Ist es auch ganz
    gewi�? -- irrst du auch nicht, Kleiner? T�usche mich nicht, indem du
    Hoffnungen in mir erweckst, die am Ende nicht in Erf�llung gehen.�

    �Das m�cht' ich um keinen Preis, Sir�, erwiderte Oliver. �Sie k�nnen
    mir in der Tat glauben. Mr. Losbernes Worte waren, sie w�rde leben
    und uns alle noch viele Jahre begl�cken. Ich hab' es ihn selbst sagen
    h�ren.�

    In seinen Augen standen Tr�nen, w�hrend er sich an die Worte erinnerte,
    die ihn so unaussprechlich gl�cklich gemacht hatten, und der junge Herr
    wandte das Gesicht ab und war einige Minuten stumm. Oliver glaubte ihn
    schluchzen zu h�ren und wagte es nicht, seinen Bericht fortzusetzen; er
    stand da und tat, als wenn er mit seinem Blumenstrau� besch�ftigt w�re.

    Mr. Giles hatte unterdes auf dem Kutschtritte, die Ellenbogen auf die
    Knie gest�tzt und die Augen trocknend, gesessen, und die R�te der
    letzteren, als der junge Herr ihn anredete, und als er aufblickte,
    bewies, da� seine Bewegung keine erk�nstelte war.

    �Fahren Sie nach meiner Mutter Hause, Giles�, sagte der junge Herr.
    �Ich will langsam nachkommen, um mich erst ein wenig zu sammeln, bevor
    ich ihr unter die Augen trete. Sie k�nnen ihr sagen, da� ich k�me.�

    �Bitt' um Vergebung, Mr. Harry,� erwiderte Giles, �aber Sie w�rden mir
    einen gro�en Gefallen erzeigen, wenn Sie sich durch den Postillon
    anmelden lassen wollten. Die Damen d�rfen mich wirklich so nicht sehen,
    Sir; ich w�rde alles Ansehen bei ihnen verlieren.�

    �Nach Ihrem Belieben, Giles�, entgegnete der junge Herr l�chelnd.
    �Lassen Sie ihn mit dem Gep�ck vorausfahren, und Sie k�nnen mit uns
    nachfolgen, nur vertauschen Sie jetzt sogleich Ihre Nachtm�tze mit
    einer angemessenen Kopfbedeckung, damit wir nicht f�r Wahnwitzige
    gehalten werden.�

    Giles erinnerte sich mit Schrecken seines unziemlichen Aufzugs, steckte
    seine Nachtm�tze in die Tasche, setzte statt derselben einen Hut auf,
    der Postillon fuhr weiter, und Giles, Mr. Maylie und Oliver folgten zu
    Fu� nach.

    Oliver blickte den jungen Herrn von Zeit zu Zeit mit ebensoviel Neugier
    wie Interesse von der Seite an. Mr. Maylie schien etwa f�nfundzwanzig
    Jahre alt zu sein, und war von Mittelgr��e; in seinem wohlgeformten
    Gesicht dr�ckte sich Offenheit aus, und sein Benehmen war �u�erst
    gewandt und gewinnend. Trotz der Altersverschiedenheit sah er der alten
    Dame so sprechend �hnlich, da� ihn Oliver sogleich als einen nahen
    Anverwandten derselben erkannt haben w�rde, wenn er sie auch nicht
    seine Mutter genannt h�tte.

    Mrs. Maylie erwartete ihn mit gro�er Sehnsucht und Ungeduld, und das
    Wiedersehen der Mutter und des Sohnes fand nicht ohne Bewegung statt.

    �O Mutter, warum schrieben Sie mir nicht fr�her?� fl�sterte er.

    �Ich schrieb allerdings,� erwiderte sie, �beschlo� aber nach reiflicher
    �berlegung, den Brief zur�ckzuhalten, bis ich Mr. Losbernes Ausspruch
    geh�rt haben w�rde.�

    �Aber warum setzten Sie sich einer Gefahr aus, deren Eintreten so sehr
    m�glich war? Wenn Rose -- ich kann das Wort jetzt nicht aussprechen --
    wenn Roses Krankheit eine andere Wendung genommen, wie h�tten Sie sich
    jemals selbst verzeihen k�nnen -- wie h�tte ich je wieder ruhig werden
    sollen?�

    �Wenn das Schlimmste eingetreten w�re, Harry, so f�rchte ich, da�
    deine Ruhe sehr wesentlich gest�rt worden und da� es von nur sehr
    geringer Bedeutung gewesen sein w�rde, ob du hier einen Tag fr�her oder
    sp�ter eingetroffen w�rest.�

    �Sie m�ssen es am besten wissen, und jedenfalls leidet das keinen
    Zweifel, da� meine Ruhe, wenn das Schlimmste eingetreten w�re --�

    �Rose verdient die echteste, reinste Neigung, die das Herz eines
    Mannes nur bieten kann. Ihr Seelenadel und ihr liebendes, hingebendes
    Gem�t rechtfertigen den Anspruch auf eine nicht gew�hnliche, sondern
    tiefe und dauernde Gegenliebe. Wenn ich davon nicht �berzeugt w�re und
    nicht au�erdem w��te, da� ein ver�ndertes Benehmen von seiten eines
    Anverwandten, den sie liebt, sie bis zum Tode betr�ben w�rde, so w�rde
    mir meine Aufgabe nicht so schwierig erscheinen, oder ich h�tte nicht
    so viele K�mpfe mit mir selbst zu bestehen, indem ich tue, was mir die
    Pflicht schlechterdings zu gebieten scheint.�

    �Ist das nicht unrecht, Mutter? Halten Sie mich noch f�r so jung,
    da� ich mein Herz nicht kennte, imstande w�re, meine innersten,
    lebhaftesten, besten Gef�hle zu mi�deuten?�

    �Mein lieber Harry, die Jugend hegt viele edle Gef�hle, welche nicht
    von Dauer und bisweilen, wenn befriedigt, um so fl�chtiger sind.
    Und was noch mehr ist, mein Sohn: -- besitzt ein enthusiastischer,
    feuriger, ehrgeiziger, junger Mann eine Gattin, auf deren Namen ein
    Flecken haftet, der, obwohl nicht ihre Schuld, von kalten und gemein
    denkenden Leuten ihr und vielleicht auch ihren Kindern, und zwar um so
    mehr zum Vorwurf gemacht wird -- um deswillen sie wie er um so mehr
    Spott und Hohn zu erdulden haben -- je erfolgreicher oder gl�nzender
    seine Laufbahn ist, so kann ihn -- und wenn er noch so gut und edel
    ist -- im sp�teren Leben die Verbindung reuen, die er in seiner Jugend
    geschlossen, und sie selbst den Schmerz und die Pein erfahren, es zu
    wissen.�

    �Mutter,� entgegnete der junge Mann ungeduldig, �ein solcher Mann w�re
    ein elender Egoist, unw�rdig des Namens eines Mannes und einer Frau,
    wie Sie sie geschildert haben.�

    �So denkst du jetzt, Harry!�

    �Und ich werde stets so denken! Die Herzensqual, die ich in den
    beiden letzten Tagen erduldet, dringt mir das offene Gest�ndnis einer
    Leidenschaft ab, die, wie Ihnen wohl bekannt, weder von gestern, noch
    eine jugendlich-leichtsinnige und unbedachte ist. Meine Neigung zu dem
    lieben, herrlichen M�dchen ist so tief und fest begr�ndet, wie es die
    Neigung eines Mannes nur sein kann. Ich habe keinen Gedanken, keinen
    Lebensplan, keine Hoffnung au�er ihr, h�her als sie, und wenn Sie sich
    meiner Liebe zu ihr widersetzen, so vernichten Sie meine Ruhe, mein
    ganzes Gl�ck f�r immer. O Mutter, �berlegen Sie noch einmal und denken
    Sie besser von mir; mi�achten Sie die hei�en Gef�hle nicht, auf welche
    Sie einen so geringen Wert zu legen scheinen.�

    �Harry,� entgegnete Mrs. Maylie, �ich halte vielmehr so viel von warmen
    und gef�hlvollen Herzen, da� ich ihnen eine Entt�uschung ersparen
    m�chte. Doch wir haben f�r jetzt genug und mehr als genug von der Sache
    geredet.�

    �So �berlassen Sie Rose die Entscheidung; und Sie werden sicher Ihren
    zu strengen Ansichten nicht so viel Macht einr�umen, da� Sie mir
    Hindernisse in den Weg legen.�

    �Das nicht; allein ich w�nsche, da� du wohl �berlegst --�

    �Ich habe �berlegt -- jahrelang �berlegt -- fast so lange, wie ich mit
    Ernst zu �berlegen f�hig bin. Meine Gef�hle sind unver�ndert geblieben
    -- werden stets unver�ndert bleiben, und warum sollte ich die Pein des
    Aufschiebens und Wartens erdulden, was ja schlechterdings keinen Nutzen
    haben kann. Ja, Rose mu� mich anh�ren, bevor ich wieder abreise!�

    �Sie soll es�, sagte Mrs. Maylie.

    �Ihr Ton scheint fast anzudeuten, da� sie mich kalt anh�ren wird,
    Mutter�, sagte der junge Mann angstvoll.

    �Nichts weniger als dies�, erwiderte die alte Dame; �weit entfernt
    davon.�

    �Hat sie auch wirklich keine andere Neigung?�

    �Nein; ich m��te sehr irren, wenn du ihr Herz nicht bereits in nur zu
    hohem Ma�e bes��est. -- H�re mich an,� fuhr sie fort, als ihr Sohn im
    Begriff stand, zu antworten; �ich will nur noch dieses sagen. Bedenke,
    ehe du dein Alles auf diesen Wurf setzest, ehe du dich zur h�chsten
    Hoffnungsstufe emportragen l�ssest, bedenke Roses Lebensgeschichte,
    mein lieber Sohn, und �berlege, welche Wirkung es auf ihre Entscheidung
    haben kann, wenn sie von ihrer zweifelhaften Herkunft in Kenntnis
    gesetzt wird; -- denn sie ist uns mit aller Innigkeit ihres edlen
    Gem�ts ergeben, und die vollkommenste Selbstaufopferung in gro�en wie
    in geringen Dingen bezeichnete stets ihre Denkart.�

    �Was wollen Sie damit sagen, Mutter?� fragte der junge Mann.

    �Ich will es dir zu erraten �berlassen�, versetzte Mrs. Maylie. �Ich
    mu� wieder zu Rose gehen. Gott sei mit dir!�

    �Werde ich Sie heute abend noch wiedersehen?�

    �Ja, sobald ich Rose verlasse.�

    �Werden Sie ihr sagen, da� ich hier bin?�

    �Nat�rlich.�

    �Und auch, welche Herzensangst ich um ihretwillen ausgestanden und wie
    mich verlangt, sie zu sehen? Sie werden mir diesen Liebesdienst nicht
    verweigern?�

    �Nein, auch das will ich ihr sagen�, erwiderte Mrs. Maylie, dr�ckte dem
    Sohne z�rtlich die Hand und ging.

    Losberne und Oliver hatten w�hrend dieser fl�chtigen Unterredung am
    fernsten Ende des Zimmers geweilt. Der erstere begr��te jetzt Harry
    Maylie auf das herzlichste und mu�te ihm sofort den umst�ndlichsten
    Bericht �ber die Krankheit und das Befinden der Patientin erstatten.
    Giles h�rte mit begierigem Ohre zu, w�hrend er mit dem Gep�ck
    besch�ftigt war.

    �Haben Sie k�rzlich etwas Besonderes geschossen, Giles?� fragte der
    Doktor nach dem Schlusse seiner Mitteilungen.

    �Nein, Sir, Besonderes eben nicht�, erwiderte Giles, hoch err�tend.

    �Auch keine Diebe gefangen oder R�uber ausfindig gemacht?� fuhr
    Losberne ein wenig boshaft fort.

    �Nein, Sir�, antwortete Giles sehr ernst.

    �Das tut mir leid, da Sie sich auf dergleichen so vortrefflich
    verstehen. Wie geht es denn Brittles?�

    �Der junge Mensch befindet sich sehr wohl, und l��t sich Ihnen ganz
    gehorsamst empfehlen, Sir.�

    �Sch�n�, sagte der Doktor. �Doch da ich Sie hier treffe, f�llt mir's
    ein, Giles, da� ich in den Tagen, wo ich so eilig abgerufen wurde,
    aufgefordert von Ihrer g�tigen Herrschaft, einen kleinen Auftrag zu
    Ihren Gunsten �bernahm. Treten Sie doch auf einen Augenblick mit mir an
    das Fenster!�

    Giles trat ziemlich verwundert zu ihm, und der Doktor beehrte ihn mit
    einer kurzen, heimlichen Unterredung, nach deren Beendigung er eine
    gro�e Menge Verbeugungen machte, und mit ungew�hnlicher Wichtigkeit
    wieder zur�ckging. Der Gegenstand des so leise gef�hrten Gespr�chs
    wurde im Zimmer nicht bekannt gegeben, wohl aber sofort in der
    K�che; denn dahin lenkte Mr. Giles augenblicklich seine Schritte und
    verk�ndete, nachdem er sich einen Krug Ale hatte reichen lassen, da�
    es seiner Herrschaft, in Anbetracht seines mutvollen Benehmens bei
    dem Einbruche, gefallen habe, die Summe von f�nfundzwanzig Pfund in
    der Sparkasse f�r ihn niederzulegen. Die K�chin und das Hausm�dchen
    hoben die H�nde und Augen empor und meinten, da� Mr. Giles jetzt ganz
    stolz werden w�rde, worauf Mr. Giles, an seiner Hemdkrause zupfend,
    erwiderte, da� sie sich in einem gro�en Irrtume bef�nden, und da� er
    ihnen dankbar sein wollte, wenn sie, falls sie dergleichen jemals
    gewahrten, ihn aufmerksam darauf machen w�rden, da� er sich hoff�rtig
    gegen Geringere erwiese. Er verbreitete sich darauf weitl�ufig �ber
    seine Bescheidenheit und Anspruchslosigkeit, wof�r ihm gro�es Lob
    gezollt wurde, wie es bei bedeutenden Personen in solchen F�llen zu
    geschehen pflegt.

    Oben verging der Rest des Abends sehr heiter, denn der Doktor befand
    sich in der fr�hlichsten Stimmung, und so erm�det oder nachdenklich
    Harry Maylie anfangs gewesen sein mochte, er konnte der guten Laune des
    wackeren Mannes nicht widerstehen. Losberne scherzte und erz�hlte, und
    Oliver glaubte nie in seinem Leben so drollige Dinge geh�rt zu haben,
    so da� er zur gro�en Freude des Doktors fortw�hrend lachte, wie der
    Doktor selbst, und endlich auch Harry; denn auch das Gel�chter hat ja
    seine ansteckende Kraft. Mit einem Wort, sie waren so vergn�gt, wie sie
    es unter den obwaltenden Umst�nden nur irgend h�tten sein k�nnen, und
    es war sp�t geworden, als sie mit leichtem und dankerf�lltem Herzen die
    Ruhe aufsuchten, deren sie nach der Ungewi�heit und Angst, in der sie
    in den letzten Tagen geschwebt hatten, so sehr bedurften.

    Oliver ging am folgenden Morgen mit mehr Hoffnung und Freude, als er,
    wie ihm schien, seit langer Zeit gekannt hatte, an seine gew�hnliche
    Besch�ftigung. Die Betr�bnis war von seinem Antlitz wie durch Zauber
    verschwunden; es war ihm, als wenn die Blumen mit doppeltem Glanze im
    Tau funkelten, die linde Luft in den Bl�ttern lieblicher s�uselte,
    der Himmel reiner und blauer als je w�re. Das ist die Wirkung unserer
    inneren Stimmung auf unsere Anschauung des �u�eren um uns her. Die auf
    die Natur und ihre Mitmenschen blicken und wehklagen, da� alles schwarz
    und finster sei, sie haben recht; allein die d�steren Farben sind
    Widerspiegelungen ihrer gelbs�chtigen Augen und Herzen. Die wahren und
    wirklichen sind zarte Tinten, und bed�rfen eines sch�rferen Gesichts.

    Eine bemerkenswerte Beobachtung entging auch Oliver nicht, n�mlich, da�
    er seine Morgenausfl�ge nicht mehr allein zu machen brauchte. Nachdem
    ihn Harry Maylie zum erstenmal mit einer Blumenladung hatte heimkehren
    sehen, wurde er von einer solchen Leidenschaft f�r Blumen ergriffen,
    und er entwickelte so viel Geschmack im Ordnen derselben, da� er Oliver
    weit hinter sich zur�cklie�, der dagegen wu�te, wo die sch�nsten
    Blumen zu finden waren. Sie durchstreiften Tag f�r Tag die Umgegend
    miteinander, und brachten die k�stlichsten Str�u�e mit nach Hause.
    Roses Fenster wurde jetzt ge�ffnet, denn die balsamische Sommerluft
    erquickte sie, und auf der Fensterbank stand jeden Morgen ein frischer,
    mit gro�er Sorgfalt geordneter Blumenstrau�. Oliver bemerkte, da� die
    welken Blumen nie weggeworfen wurden, und da� der Doktor, wenn er durch
    den Garten ging, stets hinaufblickte und bedeutsam l�chelnd den Kopf
    hin und her wiegte. So verflossen die Tage, und Roses Herstellung ging
    rasch und gl�cklich vonstatten.

    Auch unserm Oliver verging die Zeit nicht langsam, obwohl die junge
    Dame ihr Zimmer noch nicht verlassen hatte, und obwohl es keine
    Spazierg�nge wie sonst mehr gab, ausgenommen dann und wann ganz kurze
    mit Mrs. Maylie. Er verdoppelte seinen Flei� in den Lehrstunden des
    silberhaarigen, alten Mannes, so da� ihn seine raschen Fortschritte
    fast selber wundernahmen. Eines Abends, als er seine Aufgaben f�r
    den folgenden Tag lernte, begegnete ihm ein so unerwarteter wie als
    Besorgnis erregender Vorfall.

    Das kleine Zimmer, in welchem er bei seinen B�chern zu sitzen pflegte,
    befand sich im Erdgescho�, und lag nach hinten hinaus. Das Fenster ging
    in den Garten, aus welchem man durch eine T�r auf einen eingehegten
    Wiesengrund gelangte, und aus diesem auf den Anger und in ein Geh�lz.
    Es fing an zu d�mmern, Oliver hatte flei�ig gelesen und auswendig
    gelernt, es war noch immer sehr warm, auch wohl ein wenig schw�l, und
    er schlummerte �ber einem Buche ein.

    Uns beschleicht bisweilen eine Art von Schlummer, der, w�hrend er den
    Leib gefangen h�lt, der Seele ein Halbbewu�tsein der Umgebung und
    die F�higkeit, nach Belieben umherzuschweifen, l��t. Er ist Schlaf,
    sofern eine �berw�ltigende Schwere, eine L�hmung der Willenskraft
    und eine g�nzliche Unf�higkeit, unsere Gedanken und Vorstellungen zu
    beherrschen, Schlaf genannt werden kann; dennoch aber wissen wir in
    diesem Zustande, auch wenn wir tr�umen, was um uns her vorgeht, schauen
    es, h�ren, was gesprochen wird, oder welche wirkliche Laute sonst an
    unser Ohr dringen m�gen, und Wirklichkeit und Einbildung vermischen
    sich endlich so wunderbar, da� es nachgehends fast unm�glich ist,
    sie wieder voneinander zu trennen. Es ist Tatsache, obwohl unsere
    Gef�hls- und Gesichtsorgane f�r die Zeit gleichsam tot sind, da� die im
    Schlummer uns kommenden Gedanken und die in der Einbildung geschauten
    Dinge bestimmt, und zwar wesentlich bestimmt werden durch die blo�e
    stumme Gegenwart eines wirklichen Gegenstandes, der uns, als wir die
    Augen schlossen, nicht nahe zu sein brauchte, und von dessen Herannahen
    oder Anwesenheit wir kein eigentliches Bewu�tsein haben.

    Oliver wu�te genau, da� er sich in seinem kleinen Zimmer befand, da�
    seine B�cher vor ihm auf dem Tische lagen, und da� der Abendwind in
    dem Bl�tterwerk vor dem Fenster rauschte -- und schlummerte dennoch.
    Pl�tzlich trat eine g�nzliche Umwandlung seiner Umgebung ein, die Luft
    wurde hei� und dr�ckend, und er glaubte sich unter Angst und Schrecken
    wieder im Hause des Juden zu befinden. Da sa� der f�rchterliche, alte
    Mann in dem Winkel, in welchem er zu sitzen pflegte, wies mit dem
    Finger nach ihm und fl�sterte einem anderen, neben ihm sitzenden Manne,
    der das Gesicht abgewendet hatte, etwas zu.

    �Pst! mein Lieber!� glaubte er den Juden sagen zu h�ren; �er ist's,
    ist's ohne Zweifel. Kommt -- la�t uns gehen!�

    �Meint Ihr, da� ich ihn nicht erkannte?� schien der andere zu
    antworten. �Und wenn eine Rotte von Teufeln seine Gestalt ann�hme, und
    er st�nde mitten zwischen ihnen, so w�rd's mir mein Sinn zutragen,
    welcher er w�re, und ich f�nde ihn heraus. Wenn Ihr ihn f�nfzig Schuh
    tief begr�bet und br�chtet mich �ber sein Grab, so w�rd' ich wissen,
    und wenn auch kein Merkmal oder Zeichen es andeutete, da� er darunter
    begraben l�ge. M�ge sein Fleisch und Bein verfaulen, ich w�rd's!�

    Der Mann schien die Worte in einem so t�dlichen Ha� verk�ndenden Tone
    zu sprechen, da� Oliver bebend aufschreckte.

    G�tiger Himmel, welcher Anblick war es, der ihm das stockende Blut zum
    Herzen zur�cktrieb und ihn der Stimme wie der Bewegungskraft beraubte!
    Dort -- dort am Fenster -- nur zwei Schritte von ihm entfernt -- so
    nahe, da� er ihn fast h�tte ber�hren k�nnen, ehe er zur�ckschreckte --
    stand, in das Zimmer hereinlugend, der Jude, dessen Blicke den seinigen
    begegneten, und neben ihm gewahrte Oliver denselben Mann, der ihm vor
    einiger Zeit im Hofe des Gasthauses ein solches Entsetzen eingejagt;
    und der F�rchterliche war bla� vor Wut oder Grauen oder welcher inneren
    Bewegung sonst, und seine Augen schossen drohende, zornige Blicke nach
    Oliver!

    Doch sie standen da, und Oliver sah sie nur einen einzigen, fl�chtigen
    Augenblick: dann waren sie verschwunden. Sie hatten indes ihn und
    er hatte sie erkannt, und ihr Hereinlugen nach ihm und ihre Mienen
    dr�ckten sich seinem Ged�chtnis so fest und tief ein, als wenn sie in
    Stein ausgehauen und ihm von Kindheit an stets vor Augen gewesen w�ren.
    Er stand einen Augenblick wie angewurzelt da, sprang darauf aus dem
    Fenster in den Garten, und rief laut nach Hilfe.




    35. Kapitel.

        Das Endergebnis des Abenteuers, das Oliver begegnet war, und eine
        Unterredung von ziemlicher Wichtigkeit zwischen Harry Maylie und
        Rose.


    Als die Bewohner des Hauses, veranla�t durch Olivers Rufen, in den
    Garten eilten, fanden sie ihn bleich und bebend dastehen. Er wies nach
    dem Wiesengrunde hinter dem Garten und war kaum imstande, die Worte zu
    stammeln: �Der Jude! der Jude!�

    Mr. Giles vermochte gar nicht zu fassen, was sie bedeuten sollten;
    Harry Maylie, der Olivers Geschichte von seiner Mutter geh�rt hatte,
    begriff es dagegen desto rascher.

    �Welche Richtung hat er genommen?� fragte er, zugleich einen t�chtigen
    Stock aufhebend, der zuf�llig dalag.

    Oliver wies nach der Richtung hin, in welcher er die beiden M�nner
    hatte forteilen sehen und sagte, da� er sie soeben erst aus den Augen
    verloren h�tte.

    �Dann wollen wir sie schon wieder einholen!� sagte Harry. �Folgt mir,
    und haltet euch mir so nahe, wie ihr k�nnt!�

    Er sprang bei diesen Worten �ber die Hecke und eilte so raschen
    Laufes davon, da� die anderen ihm kaum zu folgen vermochten. Nach ein
    paar Minuten gesellte sich ihnen auch Losberne, der eben von einem
    Spaziergange heimkehrte, zu und rief ihnen laut die Frage zu, was denn
    vorgefallen sei. Sie hielten erst an, um Atem zu sch�pfen, als Harry in
    das Angerst�ck einlenkte, nach welchem Oliver hingewiesen hatte, und
    sorgf�ltig den Graben und die Hecke zu durchsuchen anfing, wodurch die
    �brigen Zeit gewannen, heranzukommen und Losberne die Veranlassung der
    Jagd mitzuteilen.

    Ihr Suchen war vergeblich. Sie entdeckten nicht einmal frische
    Fu�spuren. Sie standen endlich auf einem kleinen H�gel, von welchem
    aus sie die Wiesen, Anger und Felder nach allen Richtungen weithin
    �bersehen konnten. Linker Hand lag das kleine Dorf; allein die
    Verfolgten h�tten, um es zu erreichen, in der von Oliver beschriebenen
    Richtung eine Strecke �ber den offenen Anger zur�cklegen m�ssen, die
    sie in so kurzer Zeit zur�ckzulegen schlechterdings nicht imstande
    gewesen w�ren. Nach einer anderen Seite begrenzte dichtes Geb�sch
    die Wiesen, allein es war aus dem gleichen Grunde unm�glich, da� sie
    dasselbe schon hatten gewinnen k�nnen.

    �Du mu�t getr�umt haben, Oliver�, sagte Harry Maylie, ihn beiseite
    f�hrend.

    �Nein, nein, Sir, wahrlich nicht�, erwiderte der Knabe schaudernd; �ich
    sah ihn zu deutlich -- sah beide so deutlich, wie ich Sie jetzt vor mir
    sehe.�

    �Wer war denn der andere?� fragten Harry und Losberne zugleich.

    �Derselbe Mann, von dem ich Ihnen sagte, da� ich ihn im Hofe des
    Gasthauses getroffen�, antwortete Oliver. �Wir hatten unsere Blicke
    wechselseitig aufeinander geheftet, und ich k�nnte es beschw�ren, da�
    er es war.�

    �Wei�t du gewi�, da� sie diesen Weg genommen haben?� fragte Maylie.

    �So gewi�, wie ich wei�, da� sie vor dem Fenster standen�, versicherte
    Oliver, und wies nach der Hecke zwischen dem Garten und dem
    Wiesengrunde hinunter. �Da sprang der gro�e Mann hin�ber; der Jude lief
    einige Schritte weit rechts und dr�ngte sich durch die L�cke dort.�

    Maylie und Losberne sahen Oliver und sodann einander an -- und man
    brauchte nur die eifrigen Mienen des Knaben zu beobachten, um �berzeugt
    zu sein, da� er die reine Wahrheit sagte. Indes waren immer noch
    keinerlei Spuren von M�nnern, die auf eiliger Flucht begriffen gewesen
    w�ren, in irgendwelcher Richtung zu entdecken. Das Gras war lang, aber
    nur da niedergetreten, wo die Verfolgenden es niedergetreten hatten.
    Die R�nder und Seiten der Gr�ben waren von feuchter Tonerde, allein
    an keiner Stelle wollte sich auch nur die mindeste Spur frischer
    Fu�stapfen finden.

    �Es ist h�chst auffallend�, sagte Maylie.

    �H�chst auffallend�, wiederholte Losberne. �Sogar Blathers und Duff
    w�rde der Verstand dabei stillstehen.�

    Sie suchten noch immerfort, bis es vollkommen dunkel geworden war,
    und sahen sich endlich gen�tigt, ihre Bem�hungen ohne alle Hoffnung
    auf Erfolg aufzugeben. Giles mu�te sich die beiden omin�sen M�nner so
    gut wie m�glich von Oliver beschreiben lassen und wurde darauf in die
    Bierh�user des Dorfes abgeschickt, um Nachfragen anzustellen; er kehrte
    jedoch zur�ck, ohne die mindeste Auskunft erhalten zu haben, indem
    man sich doch zum wenigsten des Juden sicher erinnert haben w�rde,
    wenn er verweilt, sich etwa einen Trunk reichen lassen oder mit jemand
    gesprochen h�tte.

    Am folgenden Morgen wurden die Nachsuchungen und Nachforschungen
    wiederholt, allein ebenso vergeblich. Am zweiten Tage ging Mr. Maylie
    mit Oliver nach dem Marktflecken, in der Hoffnung, dort etwas von
    dem Juden und seinem Begleiter zu sehen, zu h�ren oder zu erfahren;
    doch der Versuch zeigte sich nicht minder fruchtlos als alle ihm
    vorhergegangenen, und nach Verlauf einiger Tage fing die Sache an in
    Vergessenheit zu geraten.

    Rose hatte inzwischen das Krankenzimmer verlassen, konnte wieder
    ausgehen, war dem Familienkreise zur�ckgegeben und erfreute aller
    Herzen durch ihr Aussehen wie durch ihre Gegenwart.

    Allein obgleich diese gl�ckliche Ver�nderung die sichtbarste Wirkung
    auf den kleinen Kreis hatte und obgleich in Mrs. Maylies Landh�uschen
    wieder muntere Gespr�che und fr�hliches Gel�chter geh�rt wurden, so
    herrschte doch bisweilen eine sonst nicht gew�hnliche Zur�ckhaltung,
    was auch Oliver nicht entging. Mrs. Maylie und ihr Sohn entfernten
    sich oft und lange, und auf Roses Wangen waren Spuren von Tr�nen
    bemerkbar. Nachdem der Doktor einen Tag zu seiner Abreise nach Chertsey
    bestimmt hatte, lag es klar vor Augen, da� etwas vorging, wodurch der
    Seelenfriede der jungen Dame und noch jemandes gest�rt wurde.

    Als endlich Rose eines Morgens im Wohnzimmer allein war, trat Harry
    Maylie herein und bat mit einigem Stocken um die Erlaubnis, ein paar
    Worte mit ihr reden zu d�rfen.

    �Wenige, sehr wenige werden hinreichen, Rose�, sagte der junge Mann,
    sich zu ihr setzend. �Was ich dir zu sagen habe, ist dir bereits nicht
    mehr unbekannt; du kennst die s��esten Hoffnungen meines Herzens,
    obgleich du sie noch niemals aus meinem Munde vernommen hast.�

    Rose war bei seinem Eintreten erbla�t, was freilich noch als eine
    Nachwirkung ihrer Krankheit gedeutet werden konnte. Sie beugte sich
    �ber einen ihr nahestehenden Blumentopf und wartete schweigend, da� er
    fortfahren w�rde.

    �Ich -- ich h�tte schon fr�her wieder abreisen sollen�, sagte er.

    �Ich bin deiner Meinung, Harry�, erwiderte Rose. �Vergib mir, da� ich
    es sage, allein ich wollte, du h�ttest es getan.�

    �Die schrecklichsten und qu�lendsten aller Bef�rchtungen haben mich
    hergetrieben�, entgegnete der junge Mann; �die Angst und Sorge,
    das teure Wesen zu verlieren, auf das sich alle meine W�nsche und
    Hoffnungen beziehen. Du warst dem Tode nahe -- standest bebend zwischen
    Himmel und Erde. Wenn die Jugendlichen, Sch�nen und Guten durch
    Siechtum heimgesucht werden, so wenden sich ihre reinen Geister den
    ewigen Wohnungen seliger Ruhe zu, und deshalb sinken die Besten und
    Sch�nsten unseres Geschlechts so oft in der Bl�te ihrer Jugend in das
    Grab.�

    Der holden Jungfrau traten, als sie diese Worte vernahm, Tr�nen in die
    Augen, und als eine derselben auf die Blume herabtr�ufelte, �ber welche
    sie sich niedergebeugt hatte, und diese versch�nend hell in ihrem
    Kelche gl�nzte, da war es, als wenn die Erg�sse eines reinen jungen
    Herzens ihre Verwandtschaft mit den lieblichsten Kindern der Natur
    geltend machten.

    �Ein Engel,� fuhr der junge Mann leidenschaftlich fort, �ein Wesen,
    so sch�n und frei von Schuld, wie ein Engel Gottes, schwebte zwischen
    Leben und Tod. Oh, wer konnte hoffen, da� sie zu den Leiden und �ngsten
    dieser Welt zur�ckkehren w�rde, als die ferne, ihr verwandte ihrem
    Blicke schon halb ge�ffnet war! Rose, Rose! es war fast zu viel, um
    es tragen zu k�nnen, zu wissen, da� du gleich einem leisen Schatten,
    den ein Licht vom Himmel auf die Erde wirft, entschw�ndest -- keine
    Hoffnung zu haben, da� du denen erhalten w�rdest, die hier noch
    weilen, und keinen Grund zu kennen, warum du es solltest -- zu wissen,
    da� du der sch�neren Welt angeh�rtest, wohin so viele Reichbegabte in
    der Kindheit und Jugend den zeitigen Flug gerichtet -- und doch bei all
    solchen Tr�stungen zu flehen, da� du den dich Liebenden wiedergegeben
    werden m�chtest! Das waren meine Gedanken bei Tag und Nacht, und mit
    ihnen ergriff mich ein so �berw�ltigender Strom von Besorgnissen und
    �ngsten und selbsts�chtigen Schmerzen, da� du sterben und nie erfahren
    w�rdest, wie hei� ich dich liebte, da� er mir in seinen Strudeln Sinn
    und Verstand fast mit fortri�. Du genasest -- Tag f�r Tag und fast
    Stunde f�r Stunde tr�ufelten wieder Tropfen der Gesundheit aus Hygieias
    Kelche herab und vermischten sich mit dem schwachen, fast versiegten,
    z�gernd in dir umlaufenden Lebensb�chlein und schwellten es wieder
    zum vollen, raschen, munteren Hinrieseln an. Ich habe dich mit Augen,
    feucht vom hei�esten Sehnen und innerster tiefer Herzensneigung,
    zur�ckkehren sehen vom Tode zum Leben. Oh, sag' mir nicht, du
    w�nschtest, da� ich meine Liebe aufgegeben haben m�chte, denn sie hat
    mein Herz erweicht und der ganzen Menschheit ge�ffnet!�

    �Das wollte ich nicht sagen�, nahm Rose weinend das Wort; �ich w�nsche
    nur, da� du von hier fortgegangen sein m�chtest, um dich wieder hohen
    und edeln Bestrebungen -- deiner w�rdigen Bestrebungen zu widmen.�

    �Es gibt keine Bestrebung, die meiner w�rdiger -- des edelsten und
    herrlichsten Geistes w�rdiger w�re als das M�hen, ein Herz wie das
    deinige zu gewinnen�, versetzte der junge Mann, ihre Hand ergreifend.
    �Rose, meine liebe, unnennbar teure Rose, ich habe dich seit -- ja,
    seit Jahren geliebt, jugendlich hoffend und tr�umend, mein Teilchen
    Ruhm mir zu erringen und dann stolz heimzukehren und im selben sch�nen
    Augenblick dir zu sagen, da� ich das Errungene nur gesucht, um es mit
    dir zu teilen, dich zu erinnern an die vielen stummen Zeichen einer
    J�nglingsneigung, die ich dir gegeben, dir dein Err�ten dabei in das
    Ged�chtnis zur�ckzurufen und dann deine Hand wie zur Besiegelung
    eines unter uns altbestandenen, stillschweigenden Vertrags zu fordern.
    Die Zeit ist noch nicht gekommen; doch gebe ich dir jetzt, ohne Ruhm
    geerntet, ohne einen der jugendlichen Tr�ume erf�llt gesehen zu haben,
    das so lange schon dein gewesene Herz und setze mein Alles auf die
    Erwiderung, die meiner Anerbietung von dir zuteil wird.�

    �Deine Handlungsweise war immer gut und edel�, erwiderte Rose, ihre
    heftige Bewegung unterdr�ckend. �Glaubst du, da� ich weder f�hllos noch
    undankbar bin, so h�re meine Antwort.�

    �Geht sie dahin, da� ich mich bem�hen soll, dich zu verdienen, teuerste
    Rose?�

    �Dahin, da� du dich bem�hen mu�t, mich zu vergessen -- nicht als deine
    alte, liebe Gespielin, denn das w�rde mich uns�glich tief verwunden und
    schmerzen, sondern als einen Gegenstand deiner Liebe. Blick' hinaus in
    die Welt -- oh, wie viele Herzen gibt es in ihr, die du gleich stolz
    sein kannst zu gewinnen. Vertraue mir eine Leidenschaft f�r eine andere
    an, und ich will dir die wahrhafteste, w�rmste und treueste Freundin
    sein.�

    Beide schwiegen, und Rose verh�llte ihr Antlitz und lie� ihren Tr�nen
    freien Lauf. Harry hielt noch immer ihre Hand stumm in der seinigen.
    �Und deine Gr�nde, Rose�, begann er endlich mit leiser Stimme; �darf
    ich die Gr�nde wissen, die dich zu dieser Entscheidung dr�ngen?�

    �Du hast ein Recht, nach ihnen zu fragen,� erwiderte Rose, �kannst
    indes nichts sagen, was meinen Beschlu� zu �ndern verm�chte. Es ist
    eine Pflicht, die ich �ben mu�. Ich bin es andern schuldig wie mir
    selbst.�

    �Dir selbst?�

    �Ja, Harry, ich bin es mir selber schuldig, da� ich, ein verwaistes,
    verm�gensloses M�dchen mit einem Flecken auf meinem Namen, der Welt
    keinen Grund gebe, zu w�hnen, ich h�tte aus niedrigen Antrieben
    deiner ersten Leidenschaft nachgegeben und mich als ein Bleigewicht
    an deine Hoffnungen und Entw�rfe geheftet. Ich bin es dir und deinen
    Angeh�rigen schuldig, dir zu wehren, im Feuer deiner edlen Gef�hle ein
    solches Hemmnis deines Vorw�rtsschreitens in der Welt dir aufzub�rden.�

    �Wenn deine Neigungen mit deinem Pflichtgef�hl zusammenstimmen --�,
    begann Harry.

    �Das ist nicht der Fall�, unterbrach ihn Rose, tief err�tend.

    �So erwiderst du also meine Liebe?� sagte Harry. �Sage mir nur dies
    eine, nur dies eine, Rose, und lindere die Bitterkeit meiner harten
    T�uschung.�

    �Wenn ich d�rfte, ohne ihm, den ich liebte, ein schweres Leid
    zuzuf�gen,� erwiderte Rose, �so w�rde ich --�

    �So w�rdest du die Erkl�rung meiner Liebe ganz anders aufgenommen
    haben?� fiel Harry in der gr��ten Spannung ein. �O Rose, verhehle mir
    das wenigstens nicht.�

    �Nun ja�, sagte die Jungfrau. �Doch�, f�gte sie, ihre Hand der seinigen
    entziehend, hinzu, �warum diese peinliche Unterredung fortsetzen, die
    f�r mich am schmerzlichsten, wenn auch ein Quell der reinsten Freude
    ist? Denn es wird mir allerdings stets ein hohes Gl�ck gew�hren, einst
    von dir wie jetzt beachtet und geliebt zu sein, und jeder neue Triumph,
    den du im Leben erringst, wird mich mit neuer Kraft und Festigkeit
    erf�llen. Lebe wohl, Harry, denn wir d�rfen uns so nie wiedersehen,
    wenn uns auch in anderen Beziehungen die sch�nsten, innigsten Bande
    umschlingen. M�ge dir jeder Segen zuteil werden, den das Flehen eines
    treuen und aufrichtigen Herzens von dort, wo die Wahrheit thront und
    alles Wahrheit ist, auf dich herabrufen kann!�

    �Noch ein Wort, Rose�, sagte Harry. �Deine wahren, eigentlichen Gr�nde.
    La� sie mich aus deinem eigenen Munde h�ren.�

    �Deine Aussichten sind gl�nzend�, erwiderte sie mit Festigkeit. �Dir
    winken alle Ehren, zu denen bedeutende Talente und einflu�reiche
    Verbindungen zu verhelfen verm�gen. Aber deine Anverwandten und
    G�nner sind stolz, und ich will mich ihnen weder aufdr�ngen, die
    Mutter verachten, die mir das Leben gab, noch auf den Sohn der Frau,
    die Mutterstelle an mir vertrat, Unehre bringen oder schuld an der
    Vereitelung seiner Hoffnungen und Aussichten sein. Mit einem Worte,�
    fuhr sie, sich abwendend, als wenn die Festigkeit sie verlie�e,
    fort, �es klebt ein Makel an meinem Namen, wie ihn die Welt an den
    Unschuldigen nun einmal heimsucht; er soll in kein fremdes Blut
    �bergehen, sondern der Vorwurf auf mir allein haften bleiben.�

    �Noch ein Wort, teuerste Rose -- noch ein einziges Wort�, rief Harry,
    sich vor ihr niederwerfend. �W�re ich minder -- minder gl�cklich, wie
    es die Welt nennt -- w�re mir ein dunkles und stilles Los beschieden
    gewesen -- w�re ich arm, krank, hilflos -- w�rdest du mich dann auch
    zur�ckweisen, oder entspringen deine Bedenken aus meinen vermuteten
    Aussichten auf Reicht�mer und Ehren?�

    �Dr�nge mich nicht zu einer Antwort auf diese Frage�, versetzte Rose.
    �Es kann und wird keine Veranlassung kommen, sie aufzuwerfen, und es
    ist nicht recht, nicht freundlich von dir --�

    �Wenn deine Antwort lautete, wie ich es fast zu hoffen wage,�
    unterbrach Harry das bebende M�dchen, �so w�rde ein Wonnestrahl auf
    meinen einsamen Weg fallen und den d�steren Pfad vor mir erhellen.
    Wieviel kannst du durch die wenigen kurzen Worte f�r mich tun, der ich
    dich �ber alles liebe! O Rose, bei meiner gl�henden, unverg�nglichen
    Neigung -- bei allem, was ich f�r dich gelitten und nach deinem
    Ausspruche leiden soll -- beantworte mir die eine Frage!�

    �Nun wohl!� erwiderte sie; �wenn dir ein anderes Los beschieden gewesen
    w�re -- wenn du immerhin ein wenig, doch nicht so hoch �ber mir
    st�ndest, wenn ich dir bei beschr�nkten Verh�ltnissen eine Gehilfin
    und Tr�sterin sein k�nnte, statt in gl�nzenden dich nur zu hindern, zu
    hemmen und zu verdunkeln, so w�rde ich dir diese ganze Pein erspart
    haben. Ich habe jetzt alle, alle Ursache, zufrieden und gl�cklich zu
    sein, w�rde dann aber, ich bekenne es, Harry, mein Gl�ck erh�ht achten.�

    Lebhafte Erinnerungen an alte, s��e Hoffnungen, die sie als aufbl�hende
    Jungfrau lange gehegt, dr�ngten sich ihr bei diesem Gest�ndnisse auf
    und brachten Tr�nen mit, wie es alte Hoffnungen tun, wenn sie verwelkt
    vor der Seele auftauchen; allein sie schafften ihrem gepre�ten Herzen
    Erleichterung.

    �Ich kann meiner Schw�che nicht wehren, und sie best�rkt mich in meinem
    Entschlu߻, f�gte sie, dem Geliebten die Hand reichend, hinzu. �In
    Wahrheit, Harry, ich mu� dich verlassen.�

    �So bitte ich um ein Versprechen�, flehte er. �La� mich noch ein
    einziges Mal -- in einem Jahr oder vielleicht noch weit fr�her -- ein
    letztes Mal �ber diesen Gegenstand zu dir reden.�

    �Nicht um in mich zu dringen, da� ich meinen wohl�berlegten Entschlu�
    �ndere, Harry; es w�rde vergeblich sein�, erwiderte Rose mit einem
    wehm�tigen L�cheln.

    �Nein,� versetzte er, �um dich ihn wiederholen zu h�ren, wenn du ihn
    wiederholen willst. Ich will dir, was ich mein nennen mag, zu F��en
    legen und der Entscheidung, die du jetzt ausgesprochen, wenn du bei ihr
    beharrst, auf keinerlei Weise entgegentreten.�

    �Dann sei es so�, sagte Rose. �Es ist nur noch eine Bitterkeit mehr,
    und ich vermag sie sp�ter vielleicht besser zu ertragen.�

    Sie reichte ihm noch einmal die Hand; allein er dr�ckte sie an seine
    Brust, k��te ihre sch�ne Stirn und eilte hinaus.




    36. Kapitel.

        Abermals ein kurzes Kapitel, das an seiner Stelle als nicht eben
        sehr wichtig erscheinen mag, aber doch gelesen werden sollte,
        weil es das vorhergehende er�rtert, und einen Schl�ssel zum
        nachfolgenden darbietet.


    �Sie sind also entschlossen, heute morgen mit mir abzureisen?� fragte
    der Doktor, als sich Harry Maylie mit ihm und Oliver zum Fr�hst�ck
    niedersetzte. �Sie �ndern ja Ihre Entschl�sse mit jeder halben Stunde.�

    �Ich hoffe, da� Sie bald anderer Meinung sein werden�, entgegnete
    Maylie, sich ohne ersichtlichen Grund verf�rbend.

    �Ich w�nsche sehr, Ursache dazu zu bekommen,� versetzte Losberne,
    �obgleich ich bekenne, da� ich daran zweifle. Gestern morgen hatten Sie
    sehr eilfertig beschlossen, zu bleiben und als ein guter Sohn Ihre Frau
    Mutter an die Seek�ste zu begleiten; kurz vor Mittag erkl�rten Sie, da�
    Sie mir die Ehre erweisen wollten, so weit mit mir zu fahren, wie ich
    auf der Londoner Stra�e bliebe; und gegen Abend drangen Sie uns�glich
    geheimnisvoll in mich, da� ich abreisen m�chte, bevor die Damen
    aufgestanden w�ren, wovon die Folge ist, da� Oliver hier beim Fr�hst�ck
    festsitzt, w�hrend er botanisieren gehen sollte. Ist's nicht zu arg,
    Oliver?�

    �Es w�rde mich sehr betr�bt haben, Sir, nicht zu Hause gewesen zu sein,
    wenn Sie und Mr. Maylie abgereist w�ren�, antwortete Oliver.

    �Bist ein guter Junge,� sagte der Doktor, �sollst zu mir kommen, wenn
    du zur�ckgekehrt bist. Doch um ernsthaft zu reden, Harry, hat eine
    Mitteilung Ihrer hohen G�nner und Freunde Ihren Abreiseeifer bewirkt?�

    �Ich habe,� erwiderte Maylie, �seit ich hier verweile, durchaus keine
    Mitteilung von meinen G�nnern und Freunden, zu denen Sie ohne Zweifel
    meinen Onkel z�hlen, erhalten, auch ist es nicht wahrscheinlich, da�
    sich eben jetzt etwas ereignet, wodurch ich zu ihnen zu eilen mich
    gedrungen f�hlen k�nnte.�

    �Sie sind ein schnurriger Kauz�, fuhr der Doktor fort. �Indes werden
    besagte G�nner Sie bei der Wahl vor Weihnachten nat�rlich ins Parlament
    bef�rdern, und Ihre pl�tzlichen Beschlu�- und Willens�nderungen sind
    keine schlechte Vorbereitung auf das �ffentliche Leben. Ein gutes
    Trainieren ist allezeit w�nschenswert, mag das Rennen Staatsstellen,
    Ehrenbechern oder Rennpreisen gelten.�

    Harry Maylie machte eine Miene, als wenn er den Doktor leicht genug
    aus dem Felde schlagen k�nnte, begn�gte sich indes zu sagen: �Wir
    werden sehen�, und lie� den Gegenstand fallen. Kurz darauf fuhr die
    Postkutsche vor, Giles holte das Gep�ck, und Losberne war eifrig
    besch�ftigt, die letzten Reisevorkehrungen zu beaufsichtigen.

    �Ein Wort, Oliver�, sagte Harry Maylie leise.

    Oliver trat zu ihm in die Fenstervertiefung, in welcher er stand, sehr
    verwundert �ber die stille Traurigkeit und Unruhe, die er zugleich an
    ihm bemerkte.

    �Du kannst jetzt recht gut schreiben�, sagte Maylie, die Hand auf den
    Arm des Knaben legend.

    �Ziemlich�, erwiderte Oliver.

    �Ich komme vielleicht vorerst nicht wieder nach Hause und w�nsche, da�
    du mir schreibst, etwa einen Montag um den andern. Willst du?� fuhr
    Harry fort.

    �Mit Freuden, Sir!� rief Oliver �u�erst erfreut �ber den Auftrag aus.

    �Ich w�nsche von dir zu h�ren, wie -- es meiner Mutter und Mi� Maylie
    ergeht; melde mir, was f�r Spazierg�nge ihr macht, wovon ihr plaudert,
    und ob sie sich wohl befinden und recht heiter sind. Du verstehst?�

    �Vollkommen, Sir.�

    �Auch w�nsche ich, da� du ihnen nichts davon sagst; es m�chte meine
    Mutter beunruhigen, so da� sie sich bewogen f�nde, mir �fter zu
    schreiben, was immer eine gro�e Bel�stigung f�r sie ist. Also mu� es
    ein Geheimnis unter uns bleiben, und schreib mir ja alles; ich verlasse
    mich auf dich.�

    Oliver f�hlte sich hochgeehrt, versprach, was von ihm verlangt wurde,
    und Maylie sagte ihm unter vielen Versicherungen seiner Zuneigung
    Lebewohl.

    Der Doktor war bereits eingestiegen, die Dienerschaft wartete am Wagen,
    Harry warf einen fl�chtigen Blick nach Roses Fenster hinauf und stieg
    gleichfalls ein.

    �Fort, Postillon!� rief er, �und fahre, so schnell du kannst; ich werde
    heute nur zufrieden sein, wenn es wie im Fluge geht.�

    �Was f�llt Ihnen ein?� rief der Doktor; �Postillon, ich werde nur
    zufrieden sein, wenn es ganz und gar nicht im Fluge geht.�

    Die Dienerschaft sah dem Wagen nach, solange er sichtbar war,
    Rose aber, die hinter den Vorh�ngen gelauscht hatte, als Harry
    hinaufblickte, schaute noch immer in die Ferne hinaus, als sich die
    Dienerschaft schon l�ngst wieder hineinbegeben hatte.

    �Er scheint ganz heiter und zufrieden zu sein�, sagte sie endlich.
    �Ich f�rchtete, da� das Gegenteil der Fall sein k�nnte, und freue mich
    meines Irrtums.�

    Tr�nen sind Zeichen sowohl der Freude wie des Schmerzes; die aber,
    welche �ber Roses Wangen hinabtr�ufelten, w�hrend sie sinnend und
    fortw�hrend in derselben Richtung hinausschauend am Fenster sa�,
    schienen mehr Kummer als Lust zu bedeuten.




    37. Kapitel.

        In welchem der Leser, wenn er in das sechsunddrei�igste Kapitel
        zur�ckblicken will, einen im ehelichen Leben nicht selten
        hervortretenden Kontrast beobachten wird.


    Mr. Bumble sa� in seinem Wohnzimmer im Armenhause und blickte
    nachdenklich und d�ster bald in den Kamin, in welchem kein Feuer
    brannte, da es Sommer war, und der daher �de und trostlos genug aussah
    und bald noch d�sterer zu dem Leimzweige empor, der von der Decke
    herabhing und von den ihr Verderben nicht ahnenden Fliegen umschw�rmt
    wurde. Vielleicht erinnerten ihn die Tierchen an eine traurige
    Begebenheit seines eigenen Lebens.

    Auch fehlte es nicht an sonstigen Anzeichen, da� in seinen
    Angelegenheiten eine bedeutende Ver�nderung vorgegangen sein mu�te.
    Wo waren der Tressenrock und der dreieckige Hut? Er trug noch
    Kniehosen und schwarze wollene Str�mpfe -- doch es waren nicht die des
    Kirchspieldieners. Der Rock war ein anderer. Der Hut ein gew�hnlicher,
    bescheidener, runder. Mr. Bumble war nicht mehr Kirchspieldiener.

    Es gibt Bef�rderungen im Leben, die, abgesehen von den mit ihnen
    verkn�pften materiellen Vorteilen, doch noch einen ganz besonderen Wert
    und eine eigent�mliche W�rde durch das mit ihnen verkn�pfte Kost�m
    erhalten. Ein Feldmarschall hat seine Uniform, ein Bischof seinen
    Ornat, ein Richter seine gro�e Per�cke, ein Kirchspieldiener seinen
    dreieckigen Hut. Man nehme dem Richter seine Per�cke, dem Bischof
    seinen Ornat oder dem Kirchspieldiener seinen dreieckigen Hut, und was
    sind sie? Weiter nichts mehr als Menschen -- blo�e Menschen. W�rde, und
    bisweilen sogar Heiligkeit h�ngen mehr von Uniformen, Ornaten, Per�cken
    und H�ten ab, als viele Leute sich tr�umen lassen.

    Mr. Bumble hatte Mrs. Corney geehelicht und war Armenhausverwalter. Ein
    anderer Kirchspieldiener war zur Gewalt gelangt, und der dreieckige
    Hut, der Tressenrock und der Stab waren auf ihn �bergegangen.

    �Morgen sind's zwei Monate!� sagte Mr. Bumble seufzend. �Es scheint ein
    Jahrhundert zu sein.�

    Mr. Bumble wollte vielleicht sagen, da� er in dem kurzen Zeitraum von
    acht Wochen ein ganzes, gl�ckliches Leben verlebt h�tte -- allein der
    Seufzer! Es lag gar viel in ihm.

    �Ich verkaufte mich�, fuhr Bumble fort, �f�r sechs Teel�ffel, eine
    Zuckerzange, einen Milchgie�er, eine Stube voll alter M�bel und zwanzig
    Pfund Geld -- nur gar zu billig, spottwohlfeil!�

    �Wohlfeil!� t�nte ihm eine schrille Stimme ins Ohr. �Du w�rst f�r jeden
    Preis zu teuer gewesen, und der Himmel wei�, da� ich dich mehr als zu
    teuer bezahlt habe.�

    Bumble drehte sich um und blickte in das Antlitz seiner liebensw�rdigen
    Eheh�lfte, welche sein kurzes Selbstgespr�ch nur unvollkommen
    verstanden und ihre erw�hnte Bemerkung auf gut Gl�ck hingeworfen hatte.

    �Frau, sei so gut, mich anzusehen�, sagte Bumble und dachte bei sich
    selbst: �Wenn sie solch einen Blick aush�lt, so h�lt sie alles aus. Er
    hat bei den Armen niemals seinen Zweck verfehlt, und verfehlt er ihn
    bei ihr, so ist es mit meiner Macht und Gewalt vorbei.�

    Er verfehlte seinen Zweck. Mrs. Bumble wurde keineswegs durch ihn
    �berw�ltigt, sondern erwiderte ihn durch einen �u�erst ver�chtlichen,
    und verband damit obendrein ein Gel�chter, das zum wenigsten klang, als
    wenn es ihr von Herzen k�me.

    Als Bumble die unerwarteten T�ne vernahm, sah er zuerst ungl�ubig und
    dann erstaunt aus, worauf er wieder in sein Br�ten und Sinnen verfiel,
    aus welchem ihn jedoch Mrs. Bumble erweckte. �Willst du den ganzen Tag
    dasitzen und schnarchen?� fragte sie.

    �Ich denke hier so lange sitzen zu bleiben, wie es mir beliebt�,
    entgegnete er; �und obschon ich keineswegs schnarchte, so bin ich doch
    gewillt, von meinem Rechte Gebrauch zu machen und ganz nach meinem
    Gefallen zu schnarchen, zu niesen, zu lachen oder zu weinen, oder was
    mir eben sonst behagt.�

    �Von deinem Rechte!� h�hnte Mrs. Bumble mit uns�glich ver�chtlicher
    Miene.

    �Ja, von meinem Rechte-1 Es ist das Recht des Mannes, nach seinem Willen
    zu leben und zu befehlen.�

    �Und was ist denn ins Kuckucks Namen das Recht der Frau?�

    �Nach des Mannes Willen zu leben und zu gehorchen. Dein ungl�cklicher,
    erster Mann h�tte es dich lehren sollen; er w�re dann vielleicht noch
    am Leben -- und ich wollte, da� er es w�re, der gute Mann!�

    Mrs. Bumble erkannte, da� der entscheidende Augenblick gekommen war,
    und da� es galt, sich der Herrschaft ein f�r allemal zu bem�chtigen,
    oder ihr f�r immer zu entsagen. Sie sank daher auf einen Stuhl nieder,
    erkl�rte Mr. Bumble f�r einen Unmenschen mit einem Kieselherzen und
    brach in einen Tr�nenstrom aus.

    Allein Tr�nen waren es nicht, was zu Mr. Bumbles Herzen drang; es war
    wasserdicht. Den Filzh�ten gleich, welche gewaschen werden k�nnen und
    durch Regen besser werden, wurden seine Nerven durch Tr�nenschauer
    noch fester, die ihn als Zeichen der Schw�che und somit als
    stillschweigende Anerkenntnisse seiner Obergewalt erfreuten und stolz
    machten. Er blickte seine Hausfrau mit gro�er Zufriedenheit an und bat
    und munterte sie auf alle Weise auf, nur immerzu zu weinen, und nach
    besten Kr�ften, denn es sei �u�erst gesund, wie die �rzte versicherten.

    �Es erweitert die Lungen, w�scht das Gesicht rein, sch�rft die Augen
    und k�hlt ein zu hei�es Temperament ab�, sagte er; �also weine ja nur
    immerzu.� -- Nachdem er die scherzenden Worte gesprochen, griff er zu
    seinem Hute, setzte ihn kecklich auf die eine Seite, wie ein Mann, der
    seine �berlegenheit f�hlt und auf geeignete Weise zeigen will, steckte
    die H�nde in die Taschen und setzte sich stolzierenden Schritts nach
    der T�r in Bewegung.

    Mrs. Bumble hatte einen Versuch mit den Tr�nen angestellt, weil sie
    minder m�hsam waren als ein Faustangriff; indes war sie vollkommen
    bereit, eine Probe mit dem letzteren Verfahren zu machen, was Mr.
    Bumble auch nicht lange verborgen blieb.

    Die erste Kunde, welche er davon erhielt, bestand in einem dumpfen
    Schalle, welcher die unmittelbare Folge hatte, da� sein Hut an das
    �u�erste Ende des Zimmers flog. Sobald durch dieses vorl�ufige Beginnen
    sein Kopf entbl��t war, packte ihn die erfahrene Dame mit der einen
    Hand bei der Kehle und lie� mit der andern einen Hagel von Schl�gen,
    und zwar ebenso gewandt wie wirksam auf sein Haupt niederfallen.
    Hierauf brachte sie ein wenig Abwechslung in ihr Vorgehen, indem
    sie ihm das Gesicht zerkratzte und H�nde voll Haare ausraufte, und
    nachdem sie ihn nunmehr so nachdr�cklich bestraft hatte, wie sie es
    dem Vergehen nach f�r n�tig erachtete, warf sie ihn �ber einen Stuhl,
    der nicht zweckm��iger h�tte stehen k�nnen und forderte ihn auf, noch
    einmal von seinen Rechten zu sprechen, wenn er es wagen wollte.

    �La� los!� rief er in befehlendem Tone, �und mach' sogleich, da� du
    fortkommst, wenn du nicht willst, da� ich etwas Desperates tue.� Er
    stand mit den allerkl�glichsten Mienen auf, sann dar�ber nach, was
    wohl ganz desperat sein m�chte, hob seinen Hut auf und blickte nach der
    T�r.

    �Gehst du bald?� fragte Mrs. Bumble.

    �Ich gehe schon, ja doch�, erwiderte er, sich rasch nach der T�r
    zur�ckziehend; �ich beabsichtige keineswegs -- wirklich, ich gehe
    schon, Liebe -- du bist aber auch so heftig, da� ich f�rwahr --�

    Mrs. Bumble b�ckte sich in diesem Augenblick, um den in Unordnung
    geratenen Teppich wieder zurecht zu schieben, und ihr Eheherr scho�
    hinaus, ohne daran zu denken, seine Rede zu vollenden, und lie� weiland
    Mrs. Corney im ungest�rten Besitz des Schlachtfeldes. -- Mr. Bumble
    war der �berraschung erlegen und ohne Frage vollst�ndig in die Flucht
    geschlagen. Er hatte die entschiedenste Neigung zum Bramarbasieren,
    nichts konnte ihm gr��ere Freude gew�hren, als Ver�bung kleiner
    Tyrannei und Grausamkeit, und er war demnach, wie kaum gesagt zu werden
    braucht, eine Memme. Hierdurch wird indes sein Charakter keineswegs
    heruntergesetzt, da so viele Beamte, die in hoher Achtung stehen und
    h�chlich bewundert werden, die Opfer �hnlicher Schw�chen sind. Wir
    haben jene Bemerkung vielmehr zu seinen Gunsten gemacht, und um unsern
    Lesern noch mehr zu Gem�t zu f�hren, wie trefflich sich Bumble zu einem
    Beamten eignete.

    Das Ma� seiner Erniedrigung war indes noch nicht voll. Nachdem er einen
    Gang durch das ganze Haus gemacht und zum erstenmal daran gedacht
    hatte, da� die Armengesetze doch wirklich zu streng w�ren, und da�
    M�nner, die von ihren Frauen fortliefen und die Erhaltung derselben dem
    Kirchspiele aufb�rdeten, von Rechts wegen ganz und gar nicht bestraft,
    sondern vielmehr als verdiente Individuen und M�rtyrer belohnt werden
    sollten, kam er in ein Gemach, in welchem die Bewohnerinnen des
    Armenhauses besch�ftigt zu werden pflegten, das Kirchspielleinenzeug zu
    waschen, und in welchem er lautes Sprechen h�rte.

    �Hm!� sagte er, seine ganze angeborene W�rde annehmend; �zum wenigsten
    sollen diese Weiber auch fernerhin meine Rechte achten. Holla -- Blitz
    und Hagel! -- wie k�nnt ihr euch unterstehen, einen solchen L�rm zu
    machen, verw�nschtes Weibsvolk?�

    Er �ffnete mit diesen Worten die T�r, schritt hochfahrend und zornig
    hinein, nahm jedoch unmittelbar darauf die dem�tigste Miene an, denn er
    erblickte seine Hausehre. �Ich wu�te nicht, da� du hier w�rst, lieber
    Schatz�, sagte er.

    �Wu�test nicht, da� ich hier war?� fuhr sie ihn an. �Was hast du denn
    hier zu schaffen?�

    �Ich dachte, sie spr�chen zu viel, um ihre Arbeiten geh�rig verrichten
    zu k�nnen�, erwiderte er, zerstreut nach ein paar alten Frauen an
    einem Waschfasse hinblickend, die bewundernde Blicke ob der Demut des
    Armenhausverwalters wechselten.

    �Du dachtest, sie spr�chen zu viel?� sagte Mrs. Bumble. �Was geht denn
    dich das an?�

    �Ei nun, lieber Schatz --�

    �Ich frage noch einmal, was es dich angeht?�

    �Es ist wahr, du hast hier zu befehlen, lieber Schatz; ich glaubte
    aber, du w�rest eben nicht bei der Hand.�

    �Ich will dir was sagen, Bumble: wir brauchen dich hier nicht, du hast
    hier nichts verloren und steckst deine Nase viel zu gern in Dinge,
    die dich nichts angehen; machst dich bei jedermann l�cherlich und zum
    Narren und wirst ausgelacht, sobald du den R�cken wendest. Troll dich
    -- willst du, oder willst du nicht?�

    Bumble gewahrte mit folternden Gef�hlen, wie die beiden alten
    W�scherinnen wahrhaft entz�ckt miteinander kicherten und z�gerte
    einen Augenblick. Mrs. Bumble, deren Geduld bei einem Aufschube nicht
    Probe hielt, ergriff ein Gef�� mit Seifenwasser, n�herte sich ihm und
    wiederholte ihre Aufforderung, bei Strafe, im Falle des Ungehorsams,
    seine stattliche Person �bersch�ttet zu sehen.

    Was konnte er tun? Er blickte trostlos umher, schlich nach der T�r,
    und das Gekicher der W�scherinnen verwandelte sich in ein schallendes
    Gel�chter. Mehr bedurfte es nicht. Er war in ihren Augen erniedrigt,
    hatte Ehre und Ansehen sogar bei den Armen verloren, war von der H�he
    der Kirchspieldienerschaft zur tiefsten Tiefe des unter Weiberregiment
    stehenden Ehemannes heruntergesunken. �Und das alles nach zwei
    Monaten!� dachte Bumble. �Kaum vor zwei -- noch vor zwei kurzen Monaten
    war ich mein eigener Herr und gebot �ber das ganze Armenhaus, und
    jetzt!�

    Es war zu viel. Er ohrfeigte den Knaben, der ihm das Tor �ffnete (denn
    er hatte mittlerweile das Portal erreicht) und trat zerstreut auf die
    Stra�e.

    Er ging eine Zeitlang auf und ab, bis sich die erste Heftigkeit seines
    Kummers gelegt hatte. Sie lie� indes Durst zur�ck. Er schritt an vielen
    Wirtsh�usern vor�ber und stand endlich vor einem in einem Nebeng��chen
    befindlichen still, dessen Gaststube, wie er durch einen fl�chtigen
    Blick sich �berzeugte, leer war. Nur ein einziger Mann sa� darin. Es
    fing eben an stark zu regnen, und dies bestimmte ihn. Er ging hinein
    und forderte ein Glas Branntwein.

    Der im Gastzimmer sitzende Mann war gro� und schw�rzlich und hatte
    sich in einen weiten Mantel geh�llt. Er schien ein Fremder und
    ziemlich weit gewandert zu sein, denn er sah erm�det aus und hatte
    staubige Stiefel an. Er blickte Bumble, als dieser eintrat, von der
    Seite an, lie� sich aber zur Entgegnung seines Gru�es kaum zu einem
    Kopfnicken herab. Bumble besa� W�rde genug f�r zwei, trank daher
    sein Glas Branntwein mit Wasser stillschweigend, und nahm mit gro�er
    Wichtigkeit ein Zeitungsblatt zur Hand. Wie es indes unter Umst�nden
    dieser Art zu geschehen pflegt, er empfand eine starke Neigung, der
    er nicht widerstehen konnte, von Zeit zu Zeit nach dem Unbekannten
    verstohlen hin�berzublicken, worauf er stets die Augen etwas verwirrt
    wieder niedersenkte, da der Unbekannte jedesmal dasselbe tat. Seine
    Verwirrung wurde noch durch den auffallenden Ausdruck der Augen des
    letzteren vergr��ert, welche scharf und durchdringend waren, und aus
    denen finstere, argw�hnische Blicke hervorschossen, wie Bumble sie noch
    nie gesehen, und die seinen Mienen etwas h�chst Absto�endes gaben.

    Als die Blicke beider einander auf diese Weise mehrmals begegnet waren,
    brach endlich der Fremde das Stillschweigen.

    �Sahen Sie nach mir,� hub er mit tiefer, rauher Stimme an, �als Sie in
    das Fenster hereinblickten?�

    �Nicht da� ich w��te, sofern Sie nicht Mr--� Bumble unterbrach sich
    hier selbst. Er w�nschte den Namen des Fremden zu erfahren und hoffte,
    da� derselbe sich nennen w�rde.

    �Ah, Sie haben also nicht nach mir hereingeblickt,� sagte der
    Unbekannte, sp�ttisch den Mund verziehend, �denn Sie w�rden sonst
    meinen Namen kennen. Ich m�chte Ihnen raten, nicht danach zu fragen.�

    �Ich habe nichts B�ses gegen Sie im Sinn, junger Mann�, entgegnete
    Bumble, sich in die Brust werfend.

    �Und haben mir auch nichts B�ses zugef�gt�, lautete die rasche Antwort.

    Es trat wiederum Stillschweigen ein, das der Fremde nach einiger
    Zeit zum zweitenmal unterbrach. �Ich sollte meinen, da� ich Sie
    schon gesehen h�tte. Sie waren zu der Zeit anders gekleidet, und ich
    begegnete Ihnen nur auf der Stra�e, erkenne Sie aber wieder. Waren Sie
    nicht Kirchspieldiener hier im Orte?�

    Bumble bejahte nicht ohne einige Verwunderung.

    �Was sind Sie denn jetzt?�

    �Armenhausverwalter�, erwiderte Bumble langsam und mit nachdr�cklicher
    Betonung, um den Unbekannten zu verhindern, einen Ton ungeb�hrlicher
    Vertraulichkeit anzunehmen. �Armenhausverwalter, junger Mann!�

    �Sie werden sich doch ohne Zweifel noch ebensogut auf Ihren Vorteil
    verstehen wie sonst?� fuhr der Unbekannte, ihn scharf anblickend,
    fort, denn Bumble sah ihn nicht wenig erstaunt an. �Tragen Sie kein
    Bedenken, mir offen zu antworten; Sie sehen ja, da� ich Sie genau genug
    kenne.�

    �Ein verheirateter Mann�, versetzte Bumble, die Augen mit der Hand
    beschattend und den Unbekannten in offenbarer Verlegenheit von
    Kopf bis zu den F��en betrachtend, �ist nicht abgeneigter als ein
    alleinstehender, auf eine ehrliche Weise ein St�ck Geld zu verdienen.
    Die Kirchspielbeamten werden nicht so reichlich besoldet, da� sie eine
    kleine Nebeneinnahme von der Hand weisen d�rften, wenn sie sich ihnen
    auf eine anst�ndige und schickliche Weise darbietet.�

    Der Unbekannte l�chelte, nickte mit dem Kopfe, als wenn er sagen
    wollte, da� er sich in seinem Manne nicht geirrt h�tte, und klingelte.
    Der Wirt erschien, er reichte ihm Bumbles leeres Glas und befahl ihm,
    es mit starkem und hei�em Getr�nk wieder zu f�llen.

    �Sie lieben es doch so?� sagte er.

    �Nicht zu stark�, erwiderte Bumble mit einem Zartgef�hl ausdr�ckenden
    Husten.

    �Sie wissen schon, was das sagen will�, rief der Unbekannte in
    trockenem Tone dem Wirt nach, der l�chelnd verschwand und kurz darauf
    mit einem dampfenden Glase zur�ckkehrte, das Bumble das Wasser in die
    Augen trieb.

    �H�ren Sie mich nun an�, sagte der Unbekannte, sobald sie wieder allein
    waren. �Ich bin heute hierher gekommen, um Sie aufzusuchen, und als
    ich eben daran dachte, wie ich Sie treffen sollte, trieb Sie mir einer
    der Zuf�lle in den Weg, durch die der Teufel bisweilen seine Freunde
    zusammenf�hrt. Ich mu� eine Erkundigung bei Ihnen einziehen, und
    verlange Ihre M�he, so gering sie sein mag, nicht umsonst. Stecken Sie
    das als Handgeld ein.�

    Er legte ein paar Goldst�cke vor ihn auf den Tisch, und nachdem Bumble
    dieselben sorgf�ltig gepr�ft hatte, ob sie auch nicht falsch w�ren,
    und sie vergn�gt in die Tasche gesteckt hatte, fuhr der Fremde fort:
    �Denken Sie einmal zur�ck -- ja an den Winter vor zw�lf Jahren.�

    �Das ist eine lange Zeit�, sagte Bumble. �Aber schon gut. Ich denke an
    den Winter.�

    �Schauplatz das Armenhaus.�

    �Gut.�

    �Zeit die Nacht.�

    �Ja, ja.�

    �Ort das elende Loch, in welchem liederliche Weibsbilder Kindern das
    ihnen selbst oft versagte Leben geben, Kindern, die das Kirchspiel
    aufzuziehen hat, und wo sie sterbend ihre Schande verstecken.�

    �Sie meinen das W�chnerinnenzimmer�, sagte Bumble.

    �Ja. In ihm wurde ein Knabe geboren.�

    �Viele, viele Knaben�, erwiderte Bumble mit kl�glichem Kopfsch�tteln.

    �Hol' der Teufel die junge H�llenbrut!� rief der Unbekannte ungeduldig
    aus. �Ich spreche von einem, 'nem zierlich und bl��lich aussehenden
    Wichte, der bei einem Leichenbestatter in die Lehre getan wurde (ich
    wollte, da� er selbst l�ngst zu Grabe getragen w�re!) und sp�ter
    fortlief, wie man glaubte, nach London.�

    �Sie meinen Oliver -- den Oliver Twist? Ich erinnere mich seiner
    nat�rlich sehr wohl. Wir hatten keinen eigensinnigeren kleinen
    Schlingel im Hause --�

    �Ich brauche nichts von ihm zu h�ren, habe genug von ihm geh�rt -- wo
    ist die alte Hexe, die seine Mutter entband?�

    �Das ist nicht leicht zu sagen. Wo sie sich jetzt aufh�lt, da gibt's
    nichts zu tun f�r Hebammen; sie wird also wohl au�er Dienst sein.�

    �Was wollen Sie damit sagen?� fragte der Unbekannte finster.

    �Da� sie im vergangenen Winter gestorben ist.�

    Der Unbekannte sah ihn eine Zeitlang scharf an, sein Blick wurde
    darauf zerstreut, und er schien in Gedanken versunken zu sein. Es war
    zweifelhaft, ob ihm die erhaltene Kunde erfreulich oder unwillkommen
    war, endlich aber schien er freier aufzuatmen, bemerkte, es k�me wenig
    darauf an, und stand auf, um sich zu entfernen.

    Bumble besa� hinreichenden Scharfsinn, um sogleich zu gewahren, da�
    sich ihm eine Gelegenheit er�ffnet habe, Gewinn aus einem Geheimnisse
    seiner besseren H�lfte zu ziehen. Er erinnerte sich des Todes der alten
    Sally sehr wohl; war sie doch an dem Abend gestorben, an welchem er
    Mrs. Corney seinen Antrag gemacht hatte; und obgleich ihm von Frau
    Bumble noch immer nicht anvertraut worden war, was die Sterbende ihr
    allein gebeichtet, so hatte er doch genug geh�rt, um zu wissen, da�
    es sich auf etwas bezogen, das sich bei oder nach der Entbindung der
    Mutter Oliver Twists ereignet hatte. Er sagte dem Unbekannten daher mit
    geheimnisvoller Miene, da� die Alte, nach welcher er sich erkundigt,
    kurz vor ihrem Tode eine andere Frau habe zu sich rufen lassen und
    derselben Mitteilungen gemacht habe, die, wie er nicht ohne Grund
    glaube, Licht in die Sache bringen k�nnten, um welche es sich handle.

    �Wo kann ich die Frau sprechen?� fragte der Unbekannte, offenbar
    �berrascht, denn er lie� durchblicken, da� er lebhafte Bef�rchtungen
    hegte, worin dieselben auch bestehen mochten.

    �Nur durch meine Vermittlung�, erwiderte Bumble.

    �Wann?� fragte der Unbekannte in gro�er Aufregung weiter.

    �Morgen.�

    �Abends um neun Uhr�, sagte der Unbekannte und schrieb mit etwas
    zitternder Hand die Adresse eines abgelegenen Hauses auf. �Bringen
    Sie sie abends um neun Uhr zu mir. Ich brauche Ihnen nicht zu sagen,
    insgeheim, denn es ist Ihr Vorteil.�

    Er ging darauf mit Bumble zur T�r, bezahlte den Wirt, bemerkte, da� sie
    sich hier trennen m��ten, sch�rfte dem Armenhausverwalter noch einmal
    P�nktlichkeit ein und ging. Bumble sah auf die Adresse; sie hatte
    keinen Namen. Er folgte daher dem Unbekannten nach, um ihn darum zu
    befragen und ber�hrte seinen Arm.

    �Was soll das?� fuhr ihn der Unbekannte, sich rasch umdrehend, an.
    �Warum folgen Sie mir nach?�

    �Ich mu� doch wissen, nach wem ich zu fragen habe�, sagte Bumble; �darf
    ich nicht um Ihren Namen bitten?�

    �Monks!� erwiderte der Unbekannte und entfernte sich mit eiligen
    Schritten.




    38. Kapitel.

        Was sich zwischen Mr. und Mrs. Bumble und Monks bei ihrer
        n�chtlichen Zusammenkunft begab.


    Es war ein schw�ler Sommerabend; die Wolken, welche den ganzen Tag
    gedroht hatten, dehnten sich zu einer breiteren und dichteren Masse
    aus, aus welcher schon dicke Regentropfen herabfielen, und schienen
    ein heftiges Gewitter zu verk�nden, als sich Mr. und Mrs. Bumble aus
    einer der Hauptstra�en der Stadt nach einer kleinen Kolonie zerstreut
    stehender und verfallener H�user wandten, die etwa anderthalb Meilen
    entfernt sein mochten, und in einer sumpfigen Niederung am Themseufer
    erbaut waren. Sie hatten sich beide in sch�bige M�ntel eingeh�llt,
    vielleicht sowohl um sich vor dem Regen zu sch�tzen, wie um unbemerkt
    zu bleiben. Mr. Bumble trug eine Laterne, in welcher jedoch kein Licht
    brannte, und ging ein paar Schritte voran, als h�tte er -- denn der Weg
    war schmutzig -- seiner Frau den Vorteil verschaffen wollen, in seine
    breiten Fu�stapfen zu treten. Sie schritten in tiefem Stillschweigen
    dahin, Mr. Bumble sah sich bisweilen um, als wenn er sich h�tte
    �berzeugen wollen, ob Mrs. Bumble auch nachfolgte, worauf er ebensooft,
    sie hinter sich gewahrend, seine Schritte wieder beschleunigte.

    Ihr Bestimmungsort konnte keineswegs ein zweideutiger hei�en, denn
    er war l�ngst als die Wohnung von verrufenem und verwegenem Gesindel
    bekannt, das haupts�chlich von Diebst�hlen und R�ubereien lebte. Es war
    ein Haufen elender Baracken, in deren Mitte am Uferrande ein gro�es
    Geb�ude stand, das ehemals zu Fabrikzwecken der einen oder anderen Art
    gedient und den H�ttenbewohnern umher wahrscheinlich Besch�ftigung
    gegeben hatte. Es war indes seit langer Zeit verfallen, und die Ratten,
    die W�rmer und die Feuchtigkeit hatten das Pfahlwerk morsch gemacht,
    auf welchem es ruhte, so da� schon ein betr�chtlicher Teil des Ganzen
    unter das Wasser gesunken war, w�hrend der wankende und �ber den
    finsteren Strom hin�berlehnende Rest nur auf eine g�nstige Gelegenheit
    zu warten schien, dasselbe Schicksal zu teilen.

    Vor diesem Geb�ude stand das w�rdige Paar still, als eben das erste
    Rollen des entfernten Donners vernehmbar wurde, und der Regen mit
    Heftigkeit niederzust�rzen anfing.

    �Es mu� hier irgendwo sein�, sagte Bumble, auf einen Papierstreifen
    blickend, den er in der Hand hielt.

    �Wer da?� ert�nte eine Stimme von oben.

    Bumble blickte empor und sah jemanden aus dem zweiten Stockwerke
    herunterschauen.

    �Eine Minute Geduld,� rief die Stimme, �ich werde sogleich bei Ihnen
    sein.�

    �Ist das der Mann?� fragte Frau Bumble, und ihr Eheherr nickte bejahend.

    �Vergi� nicht, was ich dir gesagt habe,� fuhr die Dame fort, �und
    sprich so wenig wie nur irgend m�glich, denn du wirst uns sonst gleich
    verraten.�

    Mr. Bumble, der an dem Hause mit sehr b�nglichen Blicken emporgeschaut
    hatte, stand im Begriff, einige Zweifel auszusprechen, ob es �berhaupt
    r�tlich sei, sich noch zu dieser Stunde auf das Abenteuer einzulassen,
    als er durch Monks daran gehindert wurde, der eine kleine T�r �ffnete,
    vor welcher sie standen, und ihnen winkte hereinzutreten. �Geschwind!�
    rief er ungeduldig, und mit dem Fu�e stampfend. �Haltet mich hier nicht
    auf!�

    Frau Bumble, welche anfangs gez�gert hatte, ging keck hinein, und ihr
    Eheherr, der sich sch�mte oder f�rchtete, zur�ckzubleiben, folgte ihr
    nach, jedoch offenbar mit gro�er Unruhe und ohne jene W�rde, die ihn
    sonst stets vornehmlich zu charakterisieren pflegte.

    �Was zum Teufel stehen Sie da drau�en und lie�en sich na� regnen?�
    sagte Monks zu ihm, nachdem er die T�r wieder verriegelt hatte.

    �Wir -- wir k�hlten uns ein wenig ab�, stotterte Bumble, furchtsam
    umherblickend.

    �K�hlten sich ein wenig ab!� entgegnete Monks. �Aller Regen, der jemals
    vom Himmel herabfiel oder noch herabfallen soll, wird nicht so viel
    h�llisches Feuer ausl�schen, wie ein Mann mit sich umhertragen kann.
    Glauben Sie nicht, da� Sie sich so leicht abk�hlen k�nnen.�

    Mit diesen angenehmen Worten und mit einem finsteren, stieren Blick
    wandte sich Monks zu Frau Bumble, die, obwohl sonst nicht so leicht
    einzusch�chtern, dennoch die Augen vor ihm auf den Boden heften mu�te.
    �Ist dies die Frau?� fragte Monks.

    �Hm! Ja!� antwortete Mr. Bumble, eingedenk der Warnung seiner Gattin.

    �Sie glauben vielleicht, da� Frauen keine Geheimnisse verschweigen
    k�nnen!� nahm Frau Bumble das Wort und blickte dabei Monks wieder
    dreist und forschend an.

    �Ich wei�, da� sie allezeit eins verschweigen, bis es an den Tag
    gekommen ist�, erwiderte Monks ver�chtlich.

    �Und was ist das f�r ein Geheimnis?� fragte Frau Bumble in demselben
    zuversichtlichen Tone.

    �Der Verlust ihres guten Namens�, sagte Monks; �und ebenso f�rchte ich
    nicht, da� eine Frau ihr Geheimnis ausschwatzt, wenn das Ausschwatzen
    dahin f�hren kann, da� sie geh�ngt oder deportiert wird. Verstanden?�

    �Nein�, versetzte die Dame, sich ein wenig verf�rbend.

    �Freilich,� sagte Monks sp�ttisch, �wie k�nnten Sie mich auch
    verstehen!� Er blickte die Eheleute halb h�hnisch und halb grollend an,
    winkte ihnen abermals, ihm nachzufolgen, eilte durch das gro�e, jedoch
    niedrige Zimmer voran und wollte eben eine steile Treppe oder vielmehr
    Leiter hinaufsteigen, als der helle Glanz eines Blitzes durch die
    �ffnung herabfuhr und ein Donnerschlag erfolgte, der das gebrechliche
    Geb�ude in seinem Grunde ersch�tterte.

    �H�ren Sie!� rief er, zur�ckschreckend aus. �H�ren Sie, wie es prasselt
    und rollt, als ob es durch tausend H�hlen widerhallte, wo sich die
    Teufel davor verstecken. Fluch �ber den L�rm! Ich hasse ihn.�

    Er schwieg einige Augenblicke, entfernte pl�tzlich die H�nde von seinem
    Gesicht, und Mr. Bumble gewahrte zu seinem unaussprechlichen Schrecken,
    da� es fast kreidewei� und ganz verzerrt war.

    �Ich leide bisweilen an diesen Zuf�llen,� sagte Monks, die Best�rzung
    des Armenhausverwalters bemerkend, �und dann und wann werden sie
    durch den Donner hervorgerufen. Achten Sie nicht darauf; es ist f�r
    diesmal vor�ber.� Mit diesen Worten ging er voran, erklomm die Treppe,
    verschlo� hastig die Fensterl�den des Gemaches, in welches er das
    Ehepaar f�hrte, und lie� eine an einer Leine und einer Rolle an einem
    der Deckenbalken h�ngende Laterne herunter, die ein mattes Licht auf
    einen alten Tisch und drei an denselben gestellte St�hle warf. Als sie
    sich gesetzt hatten, sagte er: �Je eher wir zur Sache kommen, desto
    besser ist's f�r uns alle. Wei� die Frau, worauf sich unser Gesch�ft
    bezieht?�

    Die Frage war an Mr. Bumble gerichtet, allein Mrs. Bumble nahm sogleich
    das Wort und erkl�rte, da� sie mit dem Zwecke der Zusammenkunft
    vollkommen bekannt sei.

    �Er sagte, Sie w�ren bei der alten Hexe an dem Abend gewesen, da sie
    starb, und sie h�tte Ihnen etwas anvertraut --�

    �Was die Mutter des Knaben betraf, den Sie nannten�, unterbrach ihn
    Frau Bumble. �Ja, Sir.�

    �Die erste Frage�, sagte Monks, �ist die, worin bestand ihre
    Mitteilung?�

    �Das ist die zweite Frage�, bemerkte Frau Bumble mit gro�er Ruhe. �Die
    erste ist die, was wohl der Preis des Geheimnisses sein mag?�

    �Wer zum Teufel kann das sagen, ohne zu wissen, worin es besteht?�
    lautete Monks' Gegenfrage.

    �Ich bin �berzeugt, niemand besser als Sie�, antwortete Frau Bumble,
    der es, wie ihr Gatte aus hinreichender Erfahrung bezeugen konnte,
    keineswegs an Herzhaftigkeit gebrach.

    �Hm!� sagte Monks bedeutsam und mit einem begierigen und lauernden
    Blick; �handelt es sich denn um etwas Wertvolles?�

    �Vielleicht -- o ja, vielleicht�, antwortete Frau Bumble gelassen.

    �Etwas, was man ihr abnahm�, fuhr Monks eifrig fort; �etwas, was sie
    trug -- etwas, was --�

    �Sie tun am besten, wenn sie bieten�, unterbrach ihn Frau Bumble. �Ich
    habe schon geh�rt, um gewi� zu sein, da� Sie der Mann sind, f�r welchen
    mein Geheimnis Wert hat.�

    Mr. Bumble, den seine bessere H�lfte von dem Geheimnis noch nicht mehr
    hatte wissen lassen, als er gleich zu Anfang gewu�t, horchte diesem
    Zwiegespr�ch mit vorgerecktem Halse und weit aufgerissenen Augen, die
    er mit unverhohlenem Erstaunen bald auf seine Frau, bald auf Monks
    heftete, und seine Spannung nahm wom�glich noch zu, als der letztere
    ernstlich nach der Summe fragte, welche f�r die Offenbarung des
    Geheimnisses gefordert w�rde.

    �Was ist es Ihnen wert?� fragte Frau Bumble ebenso kaltbl�tig wie
    vorhin.

    �Kann sein, da� es mir nichts oder da� es mir zwanzig Pfund wert ist�,
    erwiderte Monks; �sprechen Sie und lassen Sie mich Ihre Forderung
    wissen.�

    �Legen Sie noch f�nf Pfund zu; geben Sie mir f�nfundzwanzig Pfund in
    Gold,� versetzte Frau Bumble, �und ich sage Ihnen alles, was ich wei�
    -- doch eher nicht.�

    �F�nfundzwanzig Pfund!� rief Monks, sich zur�ckbeugend, aus.

    �Ich sprach so deutlich, wie ich konnte,� entgegnete Frau Bumble, �und
    die Summe ist auch nicht bedeutend.�

    �Die Summe nicht bedeutend f�r ein erb�rmliches Geheimnis, das
    vielleicht der Rede nicht wert ist, wenn Sie es offenbart haben!� rief
    Monks ungeduldig aus; �ein Geheimnis, das seit zw�lf Jahren oder l�nger
    vergessen oder begraben gelegen hat!�

    �Solche Dinge halten sich gut und verdoppeln gleich gutem Weine h�ufig
    ihren Wert durch die Zeit�, bemerkte Frau Bumble mit der kalten
    Entschlossenheit, die sie angenommen hatte; �und was das betrifft, da�
    es begraben gewesen, so gibt es Leute, die, soviel wir wissen, noch
    zw�lftausend oder zw�lf Millionen Jahre begraben liegen k�nnen und
    endlich sonderbare Geschichten erz�hlen werden.�

    �Wie aber, wenn ich f�r nichts zahle?� fragte Monks bedenklich z�gernd.

    �Sie k�nnen mir das Geld leicht wieder abnehmen�, erwiderte die
    Dame. �Ich bin ja nur eine Frau und allein und ohne Schutz in Ihrer
    abgelegenen Wohnung.�

    �Weder allein, meine Liebe, noch ohne Schutz�, fiel Mr. Bumble mit
    vor Angst bebender Stimme ein; �ich bin auch hier, meine Liebe. Und
    au�erdem,� fuhr er z�hneklappernd fort, �und au�erdem ist Mr. Monks zu
    sehr Gentleman, um sich auch nur die mindeste Gewaltt�tigkeit gegen
    Kirchspielpersonen zu erlauben. Mr. Monks wei�, da� ich nicht mehr in
    der Bl�te der Jahre und der Kraft stehe; allein er hat geh�rt -- hat
    ohne Zweifel geh�rt, lieber Schatz, da� ich ein sehr entschlossener
    Beamter und ungew�hnlich stark bin, wenn ich Veranlassung bekomme, mich
    zusammenzunehmen. Ich brauche mich nur eben etwas zusammenzunehmen.�

    Und als Mr. Bumble so sprach, machte er einen tr�bseligen Versuch, mit
    trotziger Entschlossenheit nach seiner Laterne zu greifen, und zeigte
    deutlich durch den in allen seinen Z�gen sich malenden Schrecken, wie
    es allerdings bei ihm n�tig war, da� er sich ein wenig oder vielmehr
    recht sehr zusammennehmen mu�te, bevor er sich zu einer nur irgend
    kriegerischen Demonstration herbeilie�, ausgenommen gegen Arme oder
    andere wehrlose Personen.

    �Du bist ein Narr,� entgegnete ihm seine Eheh�lfte, �und kannst nichts
    Besseres tun als den Mund halten.�

    �Und ich werde ihm sogleich darauf schlagen, wenn er nicht leiser
    spricht�, sagte Monks zornig. �Er ist also Ihr Mann?�

    �Er mein Mann!� kicherte Frau Bumble, der Frage ausweichend.

    �Ich dachte es, als Sie beide hereinkamen�, fuhr Monks fort, den
    zornigen Blick gewahrend, den die Dame ihrem Eheherrn zuwarf. �Desto
    besser; ich trage um so weniger Bedenken, mit Leuten zu unterhandeln,
    wenn ich finde, da� sie von einem und demselben Willen beseelt sind.
    Ich meine es ernstlich -- schauen Sie hier!�

    Er zog einen Beutel aus der Tasche, z�hlte f�nfundzwanzig Sovereigns
    auf den Tisch und schob sie Frau Bumble hin.

    �Nehmen Sie,� fuhr er fort, �und lassen Sie mich nun h�ren, was Sie zu
    erz�hlen haben, sobald der verw�nschte Donnerschlag vor�ber ist, der,
    ich f�hl's, gerade �ber dem Hause loswettern wird.�

    Sobald das Donnergeroll vor�ber war, hob Monks das Gesicht vom Tische
    empor und beugte sich zu Frau Bumble hin�ber, um begierig zu h�ren, was
    sie sagen w�rde. Auch das Ehepaar lehnte sich �ber den kleinen Tisch,
    so da� die K�pfe von allen dreien sich ber�hrten. Das auf sie gerade
    herunterfallende matte Licht der h�ngenden Laterne lie� ihre Gesichter
    noch bleicher und gespenstischer erscheinen, und sie sahen um so
    unheimlicher aus, als rings umher die tiefste Finsternis sie umgab.

    �Als die Frau, die wir die alte Sally nannten, starb,� hub Frau Bumble
    fl�sternd an, �war ich mit ihr allein.�

    �War niemand dabei?� fragte Monks mit demselben hohlen Gefl�ster;
    �keine Kranke oder Verr�ckte in einem anderen Bette? -- keine Seele,
    welche h�ren, vielleicht verstehen konnte?�

    �Wir waren ganz allein�, versicherte Frau Bumble; �ich und sonst
    niemand stand an ihrem Bette, als sie im Sterben lag. Sie sprach von
    einer jungen Frauensperson, die einige Jahre zuvor einem Kinde das
    Leben gegeben h�tte, und zwar nicht blo� in demselben Zimmer, sondern
    auch in demselben Bette, in welchem die Sterbende lag.�

    �F�rwahr!� sagte Monks mit bebender Lippe und �ber seine Schulter
    blickend. �Teufel! Wie doch die Dinge zuletzt kommen k�nnen!�

    �Das Kind war dasselbe, das Sie ihm gestern abend nannten�, fuhr Frau
    Bumble, nachl�ssig nach ihrem Manne hindeutend, fort; �und die alte
    Sally hat seine Mutter bestohlen.�

    �Bei ihren Lebzeiten?� fragte Monks.

    �Nein, als sie gestorben war�, erwiderte Frau Bumble mit einigem
    Schaudern. �Sie bestahl die Leiche, nachdem dieselbe eine solche
    geworden war, und was sie nahm, war eben das, was die sterbende Mutter
    in ihren letzten Atemz�gen sie gebeten hatte, um des Kindes willen
    aufzubewahren.�

    �Verkaufte sie es?� fiel Monks in der gr��ten Spannung ein; �hat sie es
    verkauft? -- Wo? -- Wann? -- An wen? -- Vor wie langer Zeit?�

    �Als sie mir mit gro�er M�he gesagt hatte, was sie getan, sank sie
    zur�ck und starb.�

    �Und sagte weiter nichts mehr?� rief Monks mit einer Stimme, die nur
    um so w�tender ert�nte, je gewaltsamer er sie zu d�mpfen suchte. �Es
    ist eine L�ge! Ich werde mich nicht hinter das Licht f�hren lassen. Sie
    sagte mehr -- ich morde Sie beide, wenn ich nicht erfahre, was es war!�

    �Sie sagte kein Sterbensw�rtchen mehr,� entgegnete Frau Bumble, allem
    Anscheine nach durch Monks' Heftigkeit nicht im mindesten erschreckt,
    was ihr Mann augenscheinlich in einem desto h�heren Grade war; �sie
    fa�te aber krampfhaft mit der einen Hand mein Kleid, und ich fand,
    als sie tot war, und als ich ihre Hand mit Gewalt losmachte, einen
    schmutzigen Papierstreifen darin.�

    �Was enthielt er?� unterbrach Monks, sich vorbeugend.

    �Nichts; es war ein Schein von einem Pfandleiher.�

    �Wor�ber?�

    �Das werde ich Ihnen seinerzeit schon sagen. Ich mu� glauben, sie hatte
    das Geschmeide, �ber dessen Empfang der Papierstreifen ausgestellt
    war, einige Zeit aufbewahrt, um gr��eren Gewinn daraus zu ziehen, es
    sodann verpf�ndet und dem Pfandleiher jedes Jahr die Zinsen bezahlt, um
    es wieder einl�sen zu k�nnen, wenn es etwa zu einer Entdeckung f�hren
    sollte. Dies war jedoch nicht geschehen, und sie starb mit dem Scheine
    in der Hand, der nach einigen Tagen verfallen sein w�rde, und ich l�ste
    das Pfand ein, weil ich glaubte, dereinst noch einmal Nutzen daraus
    ziehen zu k�nnen.�

    �Wo haben Sie es?� fragte Monks hastig.

    �Hier ist es�, erwiderte Frau Bumble und warf eilig, als wenn sie
    froh w�re, sich davon zu befreien, ein kleines ledernes Beutelchen
    auf den Tisch; Monks bem�chtigte sich desselben begierig und �ffnete
    es mit zitternden H�nden. Es enthielt ein kleines goldenes Medaillon,
    in welchem sich zwei Haarlocken und ein einfacher goldener Trauring
    befanden.

    �Auf der Innenseite ist der Name Agnes zu lesen�, sagte Frau Bumble.
    �F�r den Zunamen ist ein Raum offen gelassen, und dann folgt das Datum
    von einem Tage in dem Jahre vor der Geburt des Kindes, das ich in
    Erfahrung gebracht habe.�

    �Und das ist alles?� fragte Monks nach einer genauen und eifrigen
    Untersuchung des kleinen Beutels.

    �Ja,� antwortete Frau Bumble, und ihr Eheherr atmete lang und tief,
    als wenn er sich freute, da� alles vor�ber w�re, ohne da� Monks die
    f�nfundzwanzig Pfund zur�ckforderte. Er fa�te jetzt so viel Mut, um
    endlich den Schwei� abzuwischen, der ihm vom Anfange der Unterredung an
    �ber die Stirn und Wangen hinabgetr�ufelt war.

    �Ich wei� nichts von der Geschichte au�er dem, was ich mutma�en kann,�
    nahm seine Frau nach einem kurzen Stillschweigen wieder das Wort, �und
    begehre auch nichts zu wissen, denn es ist sicherer. Darf ich Ihnen
    aber ein paar Fragen vorlegen?�

    �Das k�nnen Sie�, sagte Monks mit einiger Verwunderung; �ob ich aber
    antworte oder nicht, ist eine andere Frage.�

    �Was ihrer drei macht�, bemerkte Mr. Bumble, ein wenig
    Scherzhaft-Witziges einschaltend.

    �War es das, was Sie von mir zu bekommen erwarteten?� fragte die Dame.

    �Ja�, erwiderte Monks. �Die zweite Frage?� --

    �Was denken Sie damit zu tun -- kann es gegen mich gebraucht werden?�

    �Niemals,� sagte Monks, �und auch nicht gegen mich. Sehen Sie hier;
    aber bewegen Sie sich keinen Schritt vorw�rts, oder Ihr Leben ist
    keinen Strohhalm wert!� Er schob bei diesen Worten pl�tzlich den Tisch
    zur Seite und �ffnete eine gro�e Fallt�r dicht vor den F��en Mr.
    Bumbles, der sich in gr��ter Hast mehrere Schritte zur�ckzog. �Schauen
    Sie hinunter�, sagte Monks, die Laterne in die �ffnung hinablassend;
    �f�rchten Sie nichts. Ich h�tte Sie ganz unbemerkt hinunter spedieren
    k�nnen, als Sie dar�ber sa�en, wenn es meine Absicht gewesen w�re.�

    Frau Bumble trat ermutigt an die �ffnung, und sogar ihr Eheherr
    wagte es, von Neugierde getrieben, dasselbe zu tun. Das vom Regen
    angeschwollene tr�be Wasser rauschte unten so gewaltig, da� sich alle
    anderen T�ne in seinem Ger�usche verloren. Es war an der Stelle vormals
    eine Wasserm�hle gewesen, und das Pfahlwerk und die sonstigen �berreste
    derselben hielten das Wasser nur auf, um seinen Andrang und das Brausen
    noch zu verst�rken.

    �Wenn man hier eine Leiche hinunterw�rfe, wo w�rde sie morgen fr�h
    sein?� fragte Monks, die Laterne in dem finsteren Schlunde hin und her
    schwingend.

    �Zw�lf Meilen weit unten im Strome und obendrein in St�cke gerissen�,
    erwiderte Bumble, bei dem blo�en Gedanken zur�ckbebend.

    Monks nahm den kleinen Beutel, band ihn fest an ein daliegendes
    bleiernes Gewicht und warf ihn in das Wasser hinunter; man h�rte,
    wie er hineinfiel, alle drei sahen einander an und schienen freier
    aufzuatmen. Monks verschlo� die Fallt�r wieder.

    �So!� sagte er. �Wenn die See ihre Toten jemals zur�ckgibt, wie B�cher
    sagen, da� sie es werde -- so wird sie doch ihr Gold und Silber samt
    jenem Plunder f�r sich behalten. Wir haben einander nichts mehr zu
    sagen und k�nnen unserem angenehmen Zusammensein ein Ende machen.�

    �Allerdings, allerdings�, bemerkte Mr. Bumble mit gro�em Eifer.

    �Sie werden doch aber reinen Mund halten?� fragte Monks mit einem
    drohenden Blick. �F�r Ihre Frau bin ich nicht besorgt.�

    �Sie k�nnen sich auf mich verlassen, junger Mann�, antwortete Bumble,
    sich unter fortw�hrenden, unendlich h�flichen Verbeugungen der Leiter
    n�hernd. �Um jedermanns willen, und Sie wissen, auch um meinetwillen,
    Mr. Monks.�

    �Ich freue mich um Ihretwillen, Sie so sprechen zu h�ren�, entgegnete
    Monks. �Z�nden Sie Ihre Laterne an, und machen Sie sich davon, so
    schnell Sie k�nnen.�

    Diese Aufforderung kam sehr zur rechten Zeit, denn Mr. Bumble w�rde,
    wenn er sich noch einmal verbeugt und dann noch einen einzigen Schritt
    zur�ckgetan h�tte, unfehlbar hinuntergest�rzt sein. Er z�ndete seine
    Laterne an, stieg schweigend hinab, und seine Frau folgte ihm. Monks
    folgte zuletzt, nachdem er einige Augenblicke gehorcht hatte, ob sich
    auch keine anderen Laute vernehmen lie�en, als die des Wasser- und
    Regenger�usches. Sie gingen langsam und vorsichtig durch das Zimmer
    im Erdgescho�, denn Monks erschrak �ber jeden Schatten, und Bumble
    hielt seine Laterne einen Fu� �ber dem Boden und blickte fortw�hrend
    angstvoll nach versteckten Fallt�ren umher. Monks �ffnete ihnen leise
    die T�r, und das Ehepaar trat in die Finsternis hinaus, nachdem es von
    seinem geheimnisvollen Bekannten durch ein Kopfnicken Abschied genommen
    hatte.

    Sobald der Armenhausverwalter und seine Gattin fort waren, rief Monks,
    der einen un�berwindlichen Widerwillen gegen das Alleinsein zu hegen
    schien, einen Knaben, der irgendwo versteckt gewesen sein mu�te, befahl
    ihm, mit der Laterne voranzugehen, und kehrte in das Gemach zur�ck, das
    er soeben verlassen hatte.




    39. Kapitel.

        In welchem alte Bekannte auftreten und Fagin und Monks die K�pfe
        zusammenstecken.


    An dem Abende, der auf die im vorigen Kapitel erz�hlte Unterredung der
    drei wackeren Leute folgte, erwachte Sikes aus seinem Schlummer und
    fragte schlaftrunken, welche Zeit es w�re. Das Zimmer, in welchem er
    sich befand, war keins von denen, die er vor der Chertseyer Expedition
    bewohnt hatte, obgleich es sich in einem Hause nicht weit von seiner
    fr�heren Wohnung befand. Es war allem Anschein ein weit schlechteres
    Gemach und erhielt nur durch ein einziges Dachfenster Licht, das auf
    eine enge und schmutzige Gasse hinausging. Auch fehlte es nicht an
    mannigfachen anderen Anzeichen, da� Mr. Sikes zur �u�ersten D�rftigkeit
    herabgesunken war, was auch durch sein bleiches und abgemagertes
    Aussehen best�tigt wurde.

    Der Einbrecher lag auf seinem Bett, in einen gro�en wei�en Mantel
    geh�llt und mit einem Gesicht, das durch seine leichenhafte Bl�sse und
    einen mindestens eine Woche alten, stachligen, schwarzen Bart nichts
    weniger als versch�nt war. Sein Hund sa� neben dem Bett, bald seinen
    Herrn mit ernsten Augen anblickend, bald die Ohren spitzend und ein
    dumpfes Knurren aussto�end, sobald ein Ger�usch auf der Stra�e oder in
    dem unteren Teile des Hauses seine Aufmerksamkeit erregte.

    An dem Fenster mit Ausbesserung eines dem Einbrecher geh�renden alten
    Kleidungsst�ckes besch�ftigt, sa� Nancy, welche gleichfalls so bla� und
    ersch�pft von Hunger und Wachen aussah, da� man sie kaum anders als
    an der Stimme erkannt haben w�rde, als sie Sikes' Frage beantwortete.
    �Noch nicht lange sieben vor�ber�, sagte sie. �Wie befindet Ihr Euch
    heute abend, Bill?�

    �So schwach wie Wasser�, erwiderte er mit einem seiner gew�hnlichen
    Fl�che. �Komm her, reich' mir die Hand und hilf mir von dem verdammten
    Bette.�

    Sikes' Laune war durch seine Krankheit nicht freundlicher geworden,
    denn w�hrend ihn Nancy emporhob und nach einem Stuhle leitete, murmelte
    er Fl�che �ber ihr Ungeschick und schlug sie.

    �Pl�rrst du?� sagte er. �La� das Winseln bleiben! Wenn du nichts
    Besseres wei�t, so troll' dich lieber. H�rst du?�

    �Freilich h�r' ich�, antwortete das M�dchen, das Gesicht abwendend und
    sich zu einem Lachen zwingend. �Was f�llt Euch denn jetzt wieder ein,
    Bill?�

    �Hast dich 'nes Bessern besonnen?� sagte Sikes finster, die in ihrem
    Auge zitternde Tr�ne gewahrend. �Um so besser f�r dich.�

    �Oh, Ihr k�nnt heut' abend nicht schlimm gegen mich sein, Bill�,
    versetzte sie, die Hand auf seine Schulter legend.

    �Warum nicht?� fuhr er sie an.

    �Wie viele, viele N�chte,� sagte sie mit einer Regung von
    Frauenz�rtlichkeit, die sogar dem Ton ihrer Stimme eine gewisse
    Weichheit gab, -- �wie viele, viele N�chte hab' ich geduldig bei Euch
    gesessen und Euch gepflegt und gewartet, als ob Ihr ein Kind gewesen
    w�ret; und Ihr w�rdet mich sicher nicht behandelt haben, wie Ihr's eben
    tatet, wenn Ihr daran gedacht h�ttet; nicht wahr, Bill? Sprecht nur ein
    Wort -- sagt nein.�

    �Nun ja, ich h�tt's nicht getan�, sagte Sikes. �Aber Gott verdamm'
    mich, die Dirne winselt schon wieder!�

    �'s ist nichts�, seufzte Nancy, sich auf einen Stuhl werfend. �K�mmert
    Euch nur nicht um mich, und es wird bald vor�ber sein.�

    �Was wird vor�ber sein?� fragte Sikes zornig. �Was hast du jetzt wieder
    f�r Dummheiten vor? Steh' auf, mach' dir zu schaffen und bleib mit
    deinen Weiberpossen zu Haus!�

    Zu jeder anderen Zeit w�rde diese Aufforderung und der Ton, in welchem
    sie ausgesprochen wurde, die beabsichtigte Wirkung gehabt haben; allein
    Nancy war in der Tat kraftlos und ersch�pft, lie� den Kopf auf die
    Stuhllehne sinken und wurde ohnm�chtig, ehe noch Sikes die angemessenen
    Fl�che aussto�en konnte, mit welchen er unter �hnlichen Umst�nden seine
    Drohungen zu w�rzen pflegte. Er wu�te nicht recht, was er tun sollte,
    denn Nancys Ohnmachten pflegten von der heftigsten Art zu sein; er nahm
    daher seine Zuflucht zu ein wenig Gottesl�sterung und rief nach Hilfe,
    als sich das Mittel vollkommen unwirksam zeigte.

    �Was gibt es, mein Lieber?� fragte der Jude hereinblickend.

    �Kannst der Dirne nicht beispringen?� rief ihm Sikes ungeduldig zu.
    �Steh' nicht da und schwatz', gaff' mich nicht an!�

    Fagin eilte mit einem Ausrufe der Verwunderung, Nancy Beistand zu
    leisten, w�hrend Mr. John Dawkins (sonst genannt der gepfefferte
    Baldowerer), der seinem ehrw�rdigen Freunde in das Zimmer gefolgt
    war, hastig ein B�ndel niederlegte, Master Charley Bates, der dicht
    hinter ihm war, eine Flasche aus der Hand ri�, sie im Nu mit den Z�hnen
    entkorkte und der Patientin einige Tropfen daraus eingo�, jedoch erst,
    nachdem er selbst gekostet, um einen etwaigen Irrtum zu verh�ten.

    �Blase ihr 'n Bissel frische Luft mit dem Blasebalge zu, Charley,�
    sagte er, �und Ihr, Fagin, klapst ihr die H�nde, w�hrend Bill ihr die
    Kleider lockert.�

    Da alle sehr eifrig waren, besonders Master Bates, dem seine Rolle
    der k�stlichste Spa� zu sein schien, so kam Nancy nach kurzer Zeit
    allm�hlich wieder zu sich selbst, wankte nach einem Stuhle am Bett,
    verbarg ihr Gesicht in den Kissen und �berlie� es Sikes, ohne alle
    Einmischung von ihrer Seite, den Neuangekommenen seine Meinung �ber sie
    und ihr unerwartetes Erscheinen auszudr�cken.

    �Welcher b�se Wind hat Euch denn hierher geblasen?� fragte er Fagin.

    �Gar kein b�ser Wind, mein Lieber,� antwortete der Jude; �denn ein
    b�ser Wind bl�st zu niemandem Gutes, und ich habe mitgebracht etwas
    Gutes, das Ihr Euch werdet freuen zu schaun. Baldowerer, mein Lieber,
    �ffne das B�ndel und gib Bill, wof�r wir haben ausgegeben all unser
    Geld.�

    Der Gepfefferte band das B�ndel auf, und Charley Bates leerte es unter
    Lobspr�chen des Inhalts.

    �Schaut nur, Bill,� sagte der junge Herr, �solch 'ne Kaninchenpastete,
    von so zarten Tierchen, da� einem sogar die Knochen auf der Zunge
    zerschmelzen; -- und hier den pr�cht'gen Tee -- und den Zucker -- und
    das Brot -- und die frische Butter -- den Gloucesterk�s -- und vor
    allen Dingen, was sagt Ihr hierzu?�

    Er stellte bei diesen Worten eine wohlverkorkte Weinflasche auf
    den Tisch, w�hrend Dawkins aus der Flasche, die er vorhin Charley
    entrissen, dem Patienten ein Glas Branntwein einschenkte, das von
    demselben sogleich auf einen Zug geleert wurde.

    �Das wird Euch bekommen, wird Euch bekommen, Bill�, sagte der Jude,
    sich vergn�gt die H�nde reibend.

    �Bekommen?� rief Sikes aus. �Ich h�tte zwanzigmal umkommen k�nnen, eh'
    du 'nen Finger f�r mich ger�hrt h�ttest. Was soll das bedeuten, du
    falscher Schuft, da� du einen in 'nem solchen Zustande l�nger als drei
    Wochen im Stich l�ssest?�

    �H�rt, Kinder, h�rt ihn nur!� sagte der Jude achselzuckend; �h�rt, was
    er sagt, da wir kommen eben und ihm bringen alle die pr�chtigen Sachen.�

    �Die Sachen sind in ihrer Art ganz gut�, bemerkte Sikes, durch einen
    Blick nach dem wohlbesetzten Tische ein wenig bes�nftigt; �aber womit
    kannst du dich entschuldigen, da� du mich hier krank, ohne Geld und
    entbl��t von allem hast liegen lassen und dich die ganze Zeit nicht
    mehr um mich bek�mmert hast, als wenn ich nicht besser w�r' wie der
    Hund da?�

    �Ich bin gewesen aus London, mein Lieber, l�nger als eine Woche�,
    erwiderte der Jude.

    �Und wo warst du die anderen vierzehn Tage,� fragte Sikes, �wo du mich
    hast hier liegen lassen wie 'ne Ratt' in ihrem Loche?�

    �Konnt's nicht �ndern, Bill�, antwortete Fagin; �kann mich nicht
    einlassen auf die Gr�nde vor so vielen Ohren; aber, auf meine Ehre, ich
    konnt's nicht �ndern.�

    �Worauf?� schnaubte ihn Sikes mit der �u�ersten Verachtung an. �Jungen,
    schneid' mir einer von euch ein St�ck Pastete ab, da� ich den Geschmack
    von seiner Ehr' aus dem Munde los werde, oder ich ekle mich daran zu
    Tode.�

    �Seid nur nicht unwirsch, mein Lieber�, erwiderte der Jude sehr
    unterw�rfig. �Ich hab' Euch vergessen nicht, Bill; niemals, Bill.�

    �Oh, ich will selbst darauf schw�ren�, fiel Sikes mit dem bittersten
    L�cheln ein. �Du gehst deinen Gesch�ften nach, w�hrend ich hier im
    Fieber liege. Ich hab' bald dies, bald das f�r dich tun m�ssen, solang
    ich gesund und auf'n Beinen war, und hab's spottwohlfeil getan und bin
    arm dabei geblieben und h�tte sterben und verderben m�ssen, w�r' die
    Dirn' nicht gewesen.�

    �Ganz recht, Bill,� sagte der Jude, Sikes' letzte �u�erung begierig
    auffassend, �w�r' nicht gewesen die Dirne! Wer aber hat sie erzogen als
    der arme, alte Fagin, und h�ttet Ihr sie gehabt ohne mich?�

    �Er hat ganz recht�, rief Nancy aus, hastig n�herkommend. �La�t ihn
    zufrieden.�

    Nancys Erscheinen gab dem Gespr�ch eine andere Wendung, denn die Jungen
    begannen auf einen Wink des schlauen alten Juden hin ihr Branntwein
    einzuschenken, w�hrend Fagin mit Aufbietung all seines Witzes Sikes
    endlich in eine bessere Laune brachte, indem er sich stellte, als
    betrachtete er seine Drohungen als kleine, harmlose Scherze und
    au�erdem von Herzen �ber ein paar rohe Sp��e lachte, zu denen sich der
    andere, nachdem er wiederholt der Branntweinflasche zugesprochen hatte,
    herablie�.

    �Das ist alles ganz gut,� sagte Sikes endlich, �aber ich mu� heute
    abend noch Geld von dir haben.�

    �Ich habe nichts, habe gar nichts bei mir, Bill�, wandte der Jude ein.

    �Dann hast du desto mehr zu Hause,� sagte Sikes, �und ich mu� darum was
    haben.�

    �Desto mehr!� rief Fagin, die H�nde emporhebend, aus. �Ich habe nicht
    soviel, um nur --�

    �Ich wei� nicht, wieviel du hast,� unterbrach ihn Sikes, �und du magst
    es selbst wohl nicht wissen, denn es wird 'ne gute Zeit dazu geh�ren,
    es zu z�hlen; aber gleichviel, ich mu� und mu� noch heut' abend Geld
    haben.�

    �Nun gut, schon gut�, entgegnete Fagin seufzend; �so will ich den
    Baldowerer schicken.�

    �Das sollst du bleiben lassen�, sagte Sikes. �Der Gepfefferte ist ein
    gut Teil zu gepfeffert und w�rd' das Herkommen vergessen oder sich
    vom Wege verlieren oder die Schuker[AP] baldowerten ihn, so da� er
    verhindert w�r', oder was er sonst f�r Ausfl�chte ers�nne. Nancy soll
    mitgehen und 's holen, und ich will mich unterdes hinlegen und dormen.�

      [AP] Polizeidiener.

    Nach vielem Markten und Feilschen kam endlich die Abrede zustande, da�
    Sikes drei Pfund und vier Schillinge erhalten solle, worauf der Jude
    mit seinen Z�glingen ging und Sikes sich niederlegte, um die Zeit bis
    zu Nancys R�ckkehr zu verschlafen. In der Wohnung des Juden sa�en
    Toby Crackit und Mr. Chitling beim f�nfzehnten Spiele Cribbage, das
    der letztere nat�rlich samt seinem f�nfzehnten und letzten Sixpence
    verlor. Mr. Crackit schien sich ein wenig zu sch�men, mit einem jungen
    Herrn sich eingelassen zu haben, der hinsichtlich seiner Stellung und
    Geistesgaben so weit unter ihm war, g�hnte, fragte nach Sikes und griff
    zu seinem Hute.

    �Niemand hier gewesen, Toby?� fragte der Jude.

    �Kein lebendiges Bein�, antwortete Mr. Crackit, an seinem Hemdkragen
    zupfend. �Ihr m��tet eigentlich ein t�chtiges St�ck Geld zahlen,
    um mich daf�r zu belohnen, da� ich Eu'r Haus solange geh�tet. Gott
    verdamm' mich, ich bin so d�mlich wie ein Geschworener und w�re so
    fest eingeschlafen wie in Newgate, wenn mich meine Gutm�tigkeit nicht
    bewogen h�tte, mich mit dem jungen Menschen abzugeben. 's ist hier
    schauderhaft langweilig gewesen.�

    Er steckte bei diesen Worten seinen Gewinn mit einer Miene in die
    Westentasche, als wenn es im Grunde tief unter seiner W�rde w�re,
    so kleine M�nzen an sich zu nehmen, und entfernte sich mit seinem
    gew�hnlichen renommistisch-gentilen Wesen. Tom Chitling sandte ihm
    bewundernde Blicke nach und erkl�rte, da� er seinen Verlust um einer
    solchen Bekanntschaft willen f�r nichts achte. Master Bates verspottete
    ihn, worauf er Fagin zur Entscheidung aufforderte. Der Jude gab Dawkins
    und Charley einen Wink und versicherte Tom, da� er ein sehr gescheiter
    junger Mensch w�re.

    �Und ist nicht Mr. Crackit eine grandige Sinze[AQ], Fagin?� fragte Tom.

      [AQ] Gro�er Herr, Gentleman.

    �Freilich, freilich, mein Lieber.�

    �Und ist's einem nicht 'ne Ehre, mit ihm Bekanntschaft zu haben?�

    �Allerdings, mein Lieber. Die beiden sind nur eifers�chtig, weil er sie
    nicht g�nnt ihnen.�

    �Seht ihr wohl?� rief Tom triumphierend. �Er hat mich ausgezogen, ich
    kann aber hingehen und wieder was verdienen und noch mehr, sobald ich
    nur will -- nicht wahr, Fagin?�

    �Ja, ja, Tom,� erwiderte der Jude, �und je eher es geschieht, desto
    besser. Also verloren mehr keine Zeit! Baldowerer, Charley, 's ist Zeit
    f�r euch, auszugehen auf Massematten[AR] -- 's ist schon fast zehn und
    noch nichts geschafft.�

      [AR] Gesch�ft, Unternehmen.

    Der Baldowerer und Charley sagten Nancy gute Nacht und entfernten sich
    unter mannigfachen Witzen auf Tom Chitlings Kosten, dessen Benehmen
    jedoch ganz und gar nicht besonders auff�llig oder ungew�hnlich gewesen
    war; denn wie viele vortrefflich junge Gentlemen gibt es nicht, die
    einen noch weit h�heren Preis bezahlen als er, um in guter Gesellschaft
    gesehen zu werden; und wie gro� ist die Anzahl der die besagte gute
    Gesellschaft bildenden feinen und vornehmen Herren, die ihren Ruf so
    ziemlich auf dieselbe Weise begr�nden, wie der elegante Toby Crackit!

    �Nun will ich dir holen das Geld, Nancy�, sagte der Jude, als sie
    fort waren. �Das ist nur der Schl�ssel zu einem kleinen Schranke, wo
    ich aufbewahre allerhand Schnurrpfeifereien, welche gebracht haben
    die Jungens. Ich verschlie�e nie mein Geld, weil ich keins habe zu
    verschlie�en. Das Gesch�ft geht schlecht, Nancy, und ich habe keinen
    Dank davon, aber ich freue mich, das junge Volk zu sehen um mich --
    pst!� unterbrach er sich, den Schl�ssel hastig wegsteckend, �was war
    das? -- horch!�

    Nancy sa� mit untergeschlagenen Armen am Tisch, und es schien ihr
    vollkommen gleichg�ltig zu sein, ob jemand k�me oder ginge und wer das
    w�re, bis das Gemurmel einer M�nnerstimme ihr Ohr traf. Sobald sie die
    Laute vernahm, legte sie mit Blitzesschnelle ihren Hut und Schal ab und
    warf beides unter den Tisch. Gleich darauf drehte der Jude sich um, und
    sie klagte mit matter Stimme, deren Ton gar sehr gegen ihre eben erst
    bewiesene, von Fagin jedoch nicht bemerkte Hast und Heftigkeit abstach,
    �ber Hitze.

    �'s ist der Mann, den ich erwartete�, sagte der Jude fl�sternd und
    offenbar verdrie�lich �ber die Unterbrechung. �Er kommt jetzt herunter
    die Treppe. Kein Wort von dem Gelde, Kind, in seiner Gegenwart. Er
    bleibt nicht lange hier -- keine zehn Minuten, liebes Kind.� Er hielt
    den kn�chernen Zeigefinger auf die Lippen, ging mit dem Licht nach
    der T�r und legte in dem Augenblick die Hand auf den Griff, als der
    Besucher hastig eintrat. -- Es war Monks.

    �Nur eine von meinen jungen Sch�lerinnen�, sagte Fagin, als Monks, eine
    Unbekannte erblickend, zur�cktrat.

    Nancy sah gleichg�ltig nach Monks hin und wandte die Blicke darauf von
    ihm ab; als er die seinigen aber auf den Juden richtete, schaute sie
    ihn abermals verstohlen, aber so scharf und forschend an, als wenn sie
    pl�tzlich eine ganz andere geworden w�re.

    �Neuigkeiten?� fragte der Jude.

    �Gro�e.�

    �Und -- und -- gute?� fragte der Jude stockend weiter, als ob er
    f�rchtete, Monks dadurch zu reizen, da� er sich zu hoffnungsfroh zeigte.

    �Zum wenigsten keine schlechten�, erwiderte Monks l�chelnd. �Ich bin
    diesmal t�tig genug gewesen. La�t uns ein paar Worte allein reden.�

    Nancy r�ckte n�her an den Tisch heran, machte aber keine Miene, das
    Zimmer zu verlassen, obwohl sie sah, da� Monks nach ihr hindeutete.
    Der Jude, der vielleicht f�rchtete, da� sie etwas von dem Gelde sagen
    m�chte, wenn er ihr befehle, hinauszugehen, wies stumm nach oben und
    ging mit Monks hinaus.

    �Nicht wieder in das h�llische Loch, wo wir damals waren�, h�rte Nancy
    den letzteren sagen, w�hrend beide die Treppe hinaufstiegen. Der Jude
    lachte und erwiderte etwas, was sie nicht verstand. Dem Schalle der
    Fu�tritte nach schienen sie in das zweite Stockwerk hinaufzugehen.
    Sie zog rasch die Schuhe aus, horchte in der gr��ten Spannung an der
    T�r und schlich, sobald sie keinen Laut mehr vernahm, vollkommen
    ger�uschlos nach. Es mochte eine Viertelstunde verflossen sein, als sie
    ebenso leise in das Zimmer zur�ckkehrte, und gleich darauf kamen auch
    die beiden M�nner wieder die Treppe herunter. Monks entfernte sich aus
    dem Hause, und als der Jude nach einiger Zeit mit dem Gelde hereintrat,
    setzte das M�dchen eben den Hut auf, wie um sich zum Fortgehen
    anzuschicken.

    �In aller Welt, Nancy, wie bla� bist du!� rief Fagin erschreckend aus.
    �Was hast du angefangen?�

    �Nichts, das ich w��te, ausgenommen, da� ich hier wer wei� wie lange
    in dem engen Zimmer gesessen habe�, antwortete sie im gleichg�ltigsten
    Tone. �Gebt mir endlich das Geld und la�t mich fort.�

    Fagin z�hlte es ihr seufzend in die Hand, sagte ihr gute Nacht, und
    sie ging. Sobald sie sich auf der offenen Stra�e befand, setzte sie
    sich auf die Stufen vor einer Haust�r und schien, ganz bet�ubt und
    ersch�pft, au�erstande zu sein, ihren Weg fortzusetzen. Pl�tzlich
    sprang sie indes wieder auf, eilte nach einer ganz anderen Richtung
    fort, als nach der, wo Sikes Wohnung lag, beschleunigte ihre Schritte
    und lief endlich, so schnell ihre F��e sie tragen konnten. Sie mu�te
    nach einer Weile stillstehen, um Atem zu sch�pfen, schien auf einmal
    wieder zur Besinnung zu kommen und rang die H�nde und brach in Tr�nen
    aus, als ob sie sich bewu�t geworden w�re, etwas nicht tun zu k�nnen,
    was zu tun sie auf das sehnlichste w�nschte.

    Sei es, da� die Tr�nen ihr Erleichterung verschafften, oder da� sie
    erkannte, wie g�nzlich hoffnungslos ihre Lage war: genug, sie kehrte
    wieder zur�ck und eilte fast ebenso schnell nach Sikes' Wohnung, sowohl
    um die verlorene Zeit wieder einzubringen, als um gleichsam mit ihren
    st�rmisch-dr�ngenden Gedanken Schritt zu halten.

    Wenn sie noch Erregtheit verriet, als sie sich dem Diebe zeigte, so
    gewahrte er dieselbe doch nicht, sondern schlummerte wieder ein,
    nachdem er gefragt, ob sie das Geld mitgebracht habe, und eine
    bejahende Antwort erhalten hatte.

    Es war ein gl�cklicher Umstand f�r das M�dchen, da� Sikes Geld erhalten
    hatte und daher am folgenden Tage durch Essen und Trinken fast
    fortw�hrend besch�ftigt wurde, was eine so wohlt�tige Wirkung auf seine
    Stimmung �u�erte, da� er weder Zeit noch Neigung hatte, sich um sie
    und ihr Benehmen sonderlich zu bek�mmern. Seinem luchs�ugigen Freunde,
    dem Juden, w�rde es nicht entgangen sein, da� sie mit der Ausf�hrung
    irgendeines verzweifelten Entschlusses umging; allein Sikes besa�
    Fagins scharfe Beobachtungsgabe nicht, so da� Nancys ungew�hnliche
    Erregtheit und Unruhe keinen Verdacht bei ihm erweckte.

    Je n�her der Abend kam, desto gr��er wurde ihre Unruhe, und als sie
    in gespannter Erwartung neben ihm sa� und darauf wartete, da� er sich
    in den Schlaf tr�nke, wurden ihre Wangen so bla�, und es blitzte ein
    so ungew�hnliches Feuer aus ihren Augen, da� Sikes endlich aufmerksam
    darauf werden mu�te. Er war matt vom Fieber, trank hei�es Wasser zu
    seinem Branntwein, um jenes minder entz�ndlich zu machen, und hatte
    Nancy das Glas gereicht, um es zum dritten oder vierten Male von ihr
    f�llen zu lassen, als ihm ihre Bl�sse und das Feuer in ihren Augen
    zuerst auffielen. Er starrte sie an, st�tzte sich auf den Ellbogen,
    murmelte einen Fluch und sagte: �Du siehst ja wie eine Leiche aus, die
    wieder zum Leben erwacht ist. Was hast du?�

    �Was ich habe?� erwiderte sie. �Nichts. Warum seht Ihr mich so scharf
    an?�

    �Was ist das wieder f�r eine Albernheit?� fragte er, die Hand auf ihre
    Schultern legend und sie unsanft sch�ttelnd. �Was ist das? Was soll das
    bedeuten? Woran denkst du, M�dchen?�

    �An vielerlei, Bill�, erwiderte sie schaudernd und die H�nde auf die
    Augen dr�ckend. �Aber was tut's?�

    Der Ton der erzwungenen Heiterkeit, in welchem sie die letzteren Worte
    gesprochen hatte, schien auf Sikes einen st�rkeren Eindruck zu machen
    als ihr wilder und starrer Blick vorher.

    �Ich will dir was sagen�, fuhr er verdrie�lich fort. �Wenn du nicht
    vom Fieber angesteckt bist und es jetzt selbst bekommst, so ist etwas
    mehr als Gew�hnliches im Winde und obendrein was Gef�hrliches. Du
    willst doch nicht hingehen und -- nein, Gott verdamm'! das kannst du
    nimmermehr!�

    �Was kann ich nimmermehr?� fragte das M�dchen.

    �Es gibt,� murmelte Sikes, die Blicke auf sie heftend, �es gibt keine
    zuverl�ssigere, treuere Dirne in der Welt als sie, oder ich w�rde ihr
    vor drei Monaten die Kehle abgeschnitten haben. Sie kriegt das Fieber
    -- das ist das ganze.�

    Er leerte das Glas und forderte darauf seine Arznei. Nancy sprang rasch
    auf, bereitete sie, den R�cken ihm zukehrend, und gab sie ihm ein.

    �Jetzt setze dich hier an mein Bett�, sagte er, �und nimm dein eigenes
    Gesicht vor, oder ich �ndere es so, da� du es selbst nicht wieder
    erkennst, wenn du es brauchst.�

    Sie tat nach seinem Gehei�, er fa�te ihre Hand, sank auf das Kissen und
    heftete die Augen auf ihr Gesicht. Sie fielen ihm zu, er �ffnete sie
    wieder, blickte starr umher und verfiel endlich in einen tiefen und
    schweren Schlummer. Der Griff seiner Hand l�ste sich, der ausgestreckte
    Arm fiel schlaff nieder, und Sikes lag da wie in dumpfer Bet�ubung.

    �Der Schlaftrunk hat endlich gewirkt�, murmelte sie; �doch vielleicht
    ist es schon zu sp�t.�

    Sie kleidete sich hastig an, blickte furchtsam umher, als wenn sie
    trotz des Schlaftrunks jeden Augenblick erwartete, den Druck von Sikes'
    schwerer Hand auf ihrer Schulter zu f�hlen, beugte sich �ber das Bett,
    k��te den Mund des R�ubers, �ffnete und verschlo� ger�uschlos die
    T�r und eilte aus dem Hause. Ein W�chter rief halb zehn Uhr, und sie
    fragte ihn, ob es schon lange nach halb zehn w�re. Er erwiderte, eine
    Viertelstunde; sie murmelte: �und ich kann erst in einer Stunde dort
    sein�, und eilte rasch weiter.

    Sie schlug die Richtung von Spitalfields nach Westend ein. Viele der
    L�den in den engen Seitengassen, durch die sie ihr Weg f�hrte, waren
    schon geschlossen. Als es zehn schlug, wuchs ihre Unruhe, zumal da
    sie vielfach durch das Gedr�nge in den belebteren Stra�en aufgehalten
    wurde. Sie eilte so ungest�m und r�cksichtslos auf Gefahr jeder
    Art weiter, da� sie von den Fu�g�ngern f�r eine Verr�ckte gehalten
    wurde. Als sie sich Westend n�herte, nahm das Gedr�nge ab, und sie
    beschleunigte ihre Schritte noch mehr. Endlich erreichte sie ihren
    Bestimmungsort: ein sch�nes Haus in einer Stra�e nicht weit vom
    Hydepark. Es schlug eben elf. Sie trat in den Hausflur. Der Sitz des
    T�rstehers war leer; sie blickte ungewi� umher und n�herte sich der
    Treppe.

    �Zu wem wollen Sie, junges Frauenzimmer?� rief ihr ein wohlgekleidetes
    Stubenm�dchen, das eine T�r hinter ihr �ffnete, nach.

    �Zu einer Dame hier im Hause.�

    �Einer Dame!� lautete die mit einem Blicke der Verachtung begleitete
    Antwort. �Zu was f�r einer Dame?�

    �Mi� Maylie�, sagte Nancy.

    Das M�dchen, das jetzt Zeit gehabt hatte, die Fremde genauer anzusehen,
    antwortete nur durch einen Blick tugendhafter Entr�stung und rief einen
    Bedienten, dem Nancy ihre Bitte wiederholte. Er fragte nach ihrem Namen.

    �Sie brauchen gar keinen zu nennen.�

    �In was f�r 'ner Angelegenheit wollen Sie die Dame sprechen?�

    �Ich mu� sie sprechen -- das gen�gt.�

    Der Bediente befahl ihr, sich aus dem Hause zu entfernen und schob sie
    nach der T�r hin.

    �Nehmen Sie sich in acht -- Sie werden mich nicht lebendig aus dem
    Hause hinausschaffen!� rief sie. �Ist denn niemand hier, der einem
    armen M�dchen den kleinen Dienst leistet, zu der Dame hinaufzugehen?�

    Inzwischen hatte sich die Dienerschaft versammelt. Der gutm�tige Koch
    legte sich in das Mittel und forderte den Bedienten auf, das M�dchen
    Mi� Rose zu melden.

    �Wozu denn aber?� antwortete der Bediente. �Sie werden doch nicht
    glauben, da� die junge Dame eine solche Person vorlassen wird?�

    Diese Anspielung auf Nancys verd�chtigen Stand erregte ein gewaltiges
    Ma� tugendsamer Entr�stung bei vier Dienstm�dchen, welche mit
    gro�er Lebhaftigkeit erkl�rten, das Gesch�pf sei eine Schande ihres
    Geschlechts, und darauf bestanden, sie ohne Gnade auf die Stra�e zu
    werfen.

    �Machen Sie mit mir, was Ihnen beliebt,� sagte das M�dchen, zu den
    Bedienten sich wendend, �nur tun Sie erst, was ich verlange; und ich
    fordere Sie auf, meine Botschaft um Gottes willen auszurichten.�

    Der weichherzige Koch trat jetzt vermittelnd dazwischen, und das Ende
    war, da� der Mann, der zuerst zum Vorschein gekommen, die Meldung
    �bernahm.

    �Was soll ich meiner Herrschaft sagen?� fragte er.

    �Da� ein junges M�dchen Mi� Maylie unter vier Augen zu sprechen
    w�nscht�, erwiderte Nancy; �und -- da� die junge Dame, wenn sie nur
    das erste Wort anh�ren will, sogleich erkennen wird, ob sie das, was
    ich anzubringen habe, noch ferner anh�ren mu�, oder mich als eine
    Betr�gerin vor die T�r werfen lassen soll.�

    �Meiner Six!� erwiderte der Bediente. �Sie sind Ihrer Sache ja sehr
    gewi�.�

    �Bringen Sie nur mein Anliegen vor, und lassen Sie mich den Bescheid
    wissen�, entgegnete das M�dchen fest.

    Der Bediente eilte hinauf, und Nancy stand bleich, fast atemlos und mit
    zuckenden Lippen da, als die sehr h�rbaren Ausdr�cke von Verachtung ihr
    Ohr trafen, mit welchen die tugendreichen Dienstm�dchen sehr freigebig
    waren. Ihre Bl�sse nahm zu, als der Bediente wieder herunterkam und ihr
    sagte, da� sie hinaufgehen k�nne.

    �Rechtlich sein hilft zu nichts in dieser Welt�, bemerkte das erste
    Dienstm�dchen.

    �Messing hat's besser als das Gold, das die Feuerprobe bestanden hat�,
    sagte das zweite.

    Das dritte begn�gte sich damit, seine Verwunderung dar�ber
    auszusprechen: �aus welchem besseren Stoffe die Damen wohl sein
    m�chten�; und das vierte �bernahm die Sopranstimme im Quartett: �'s ist
    'ne Schande�, womit die vier Dianen schlossen.

    Ohne auf dieses alles zu achten -- denn sie hatte wichtigere Dinge
    auf dem Herzen -- folgte Nancy mit Beben dem Bedienten in ein kleines
    Vorzimmer, das durch eine von der Decke herabh�ngende Lampe erleuchtet
    war, und in welchem ihr F�hrer sie allein lie�.




    40. Kapitel.

        Eine seltsame Zusammenkunft, die eine Folge von den im vorigen
        Kapitel erz�hlten Ereignissen ist.


    Nancy hatte ihr ganzes Leben in den Stra�en und den ekelhaftesten
    H�hlen des Lasters der Hauptstadt zugebracht, dabei aber immer noch
    sich einen Rest von der Natur des Weibes bewahrt; und als sie die
    leichten, der T�r sich n�hernden Schritte vernahm und des weiten
    Abstandes der Personen gedachte, die das Gemach im n�chsten Augenblick
    einschlie�en w�rde, f�hlte sie sich durch die Last ihrer tiefen Schmach
    g�nzlich zu Boden gedr�ckt und fuhr in sich zusammen, wie wenn sie
    die Gegenwart der Dame kaum zu ertragen verm�chte, bei welcher sie
    vorgelassen zu werden gebeten hatte.

    Allein gegen die besseren Gef�hle k�mpfte der Stolz an -- die S�nde
    der Niedrigsten und Verworfensten wie der H�chststehenden und im
    Guten befestigt sich D�nkenden. Die elende Genossin von Dieben und
    B�sewichtern aller Art, die tiefgesunkene Bewohnerin der gemeinsten
    Schlupfwinkel, die Genossin der Ausw�rflinge der Gef�ngnisse und der
    Galeeren, die selbst im Galgenbereiche Lebende -- selbst diese mit
    Schmach und Schande Beladene empfand zu viel Stolz, um auch nur einen
    schwachen Schimmer des weiblichen Gef�hls zu verraten, welches ihr als
    Schw�che erschien, w�hrend es noch das einzige Band war zwischen ihr
    und der besseren Menschheit, deren �u�ere Spuren und Kennzeichen alle
    ihr w�stes Leben bei ihr vertilgt hatte.

    Sie erhob die Augen zur Gen�ge, um zu gewahren, da� die Gestalt, welche
    jetzt erschien, die eines zartgebauten, holden M�dchens war; sie senkte
    die Blicke nieder und sagte, den Kopf mit angenommener Gleichg�ltigkeit
    emporwerfend: �Es hat schwer gehalten, zu Ihnen gelassen zu werden,
    Lady. W�r' ich empfindlich gewesen und fortgegangen, wie es viele getan
    haben w�rden, Sie m�chten es dereinst bereut haben, und nicht ohne
    Grund.�

    �Es tut mir leid, wenn man Sie unartig behandelt hat�, erwiderte Rose.
    �Denken Sie nicht mehr daran und sagen Sie mir, weshalb Sie mich zu
    sprechen w�nschen.�

    Der g�tige Ton, in welchem sie antwortete, ihre freundlich klingende
    Stimme, ihr sanftes Wesen und der Umstand, da� sie gar keinen Hochmut,
    kein Mi�fallen zeigte, �berraschten Nancy dergestalt, da� sie in einen
    Tr�nenstrom ausbrach.

    �O Lady, Lady!� rief sie, die aufgehobenen H�nde leidenschaftlich
    ringend, �wenn mehrere Ihresgleichen w�ren, w�rden weniger
    meinesgleichen sein -- gewi� -- gewi�!�

    �Setzen Sie sich�, sagte Rose; �Ihre Worte gehen mir in der Tat an das
    Herz. Wenn Sie in bed�rftiger Lage oder sonst ungl�cklich sind, so werde
    ich mich gl�cklich sch�tzen, Ihnen, wenn ich es vermag, beizustehen --
    glauben Sie es mir. Setzen Sie sich.�

    �Lassen Sie mich nur stehen, Lady,� sagte das M�dchen, noch immer
    Tr�nen vergie�end, �und reden Sie nicht so g�tig zu mir, bis Sie
    mich besser kennen lernen. Doch es wird sp�t. Ist -- ist -- jene T�r
    verschlossen?�

    �Ja�, erwiderte Rose, einige Schritte zur�ckweichend, als ob sie im
    Notfalle der Hilfe nahe zu sein w�nschte. �Weshalb aber?�

    �Weil ich im Begriff bin, mein Leben und das Leben anderer in Ihre
    H�nde zu legen. Ich bin das M�dchen, das den kleinen Oliver zu Fagin,
    dem alten Juden, an jenem Abend wieder zur�ckschleppte, als er das Haus
    in Pentonville verlie�.�

    �Sie!� sagte Rose Maylie.

    �Ja, ich, Lady. Ich bin die Sch�ndliche, von der Sie ohne Zweifel
    geh�rt haben, die unter Dieben lebt und die, Gott helfe mir! solange
    ich zur�ckdenken kann, kein besseres Leben oder freundlichere Worte,
    als meine Genossen mir geben, gekannt hat. Ja, weichen Sie nur
    immerhin entsetzt vor mir zur�ck, Lady. Ich bin j�nger, als Sie nach
    meinem Aussehen glauben m�gen: allein ich bin daran gew�hnt, und die
    �rmsten Frauen entziehen sich meiner Ber�hrung, wenn ich durch die
    dichtgedr�ngten Stra�en gehe.�

    �Wie schrecklich!� sagte Rose, sich von dem M�dchen unwillk�rlich noch
    weiter entfernend.

    �Danken Sie auf Ihren Knien dem Himmel, geehrte Lady,� rief die
    Ungl�ckliche aus, �da� Sie Angeh�rige haben, die Sie in Ihrer Jugend
    bewacht und gepflegt, und da� Sie niemals wie ich seit der fr�hesten
    Kindheit, von K�lte und Hunger, von V�llerei und Trunkenheit und -- und
    von noch etwas viel Schlimmerem, als dieses alles ist, umgeben gewesen
    sind. Ich darf es sagen, denn elende Gassen und w�ste H�hlen sind meine
    Behausung gewesen und werden mein Sterbebett sein.�

    �Ich bemitleide Sie!� sagte Rose mit bebender Stimme. �Es ist ja
    herzzerrei�end, Sie anzuh�ren.�

    �Gottes Segen �ber Sie und Ihre G�te!� erwiderte das M�dchen. �Wenn
    Sie w��ten, wie es mir bisweilen zumute ist, Sie w�rden mich bedauern,
    glauben Sie mir. Doch ich habe mich fortgeschlichen von Leuten, die
    mich sicherlich ermorden w�rden, w��ten sie, da� ich hier gewesen
    bin, um Sie von Dingen, die ich ihnen abgehorcht habe, in Kenntnis zu
    setzen. Ist Ihnen ein Mensch namens Monks bekannt?� Rose verneinte.

    �Er kennt Sie�, fuhr das M�dchen fort, �und wu�te, da� Sie hier
    wohnten, denn nur dadurch, da� er es einem anderen sagte, ward es mir
    m�glich, Sie aufzufinden.�

    �Ich habe den Namen niemals nennen h�ren�, sagte Rose.

    �Nun, so f�hrt er unter uns einen anderen, was ich wohl schon fr�her
    vermutet habe. Vor einiger Zeit und bald nachdem Oliver in der Nacht
    des beabsichtigten Raubes in Ihr Haus gehoben wurde, behorchte ich
    diesen Menschen, auf welchen ich Verdacht geworfen, als er mit Fagin
    eine Unterredung hatte. Ich erfuhr bei der Gelegenheit, da� Monks --
    der Mann, nach welchem ich Sie vorhin fragte --�

    �Wohl, ich verstehe schon�, sagte Rose.

    �Da� Monks den Knaben an eben dem Tage, als wir ihn verloren, mit zwei
    von unseren Knaben zuf�llig erblickte und sogleich in ihm das Kind
    erkannt hatte, welchem er auflauerte, wiewohl ich mir nicht erkl�ren
    konnte, weshalb. Er wurde mit Fagin dar�ber einig, da� der Jude,
    falls Oliver wieder zur�ckgebracht w�rde, eine gewisse Summe und noch
    mehr erhalten solle, wenn er einen Dieb aus ihm machte, was Monks zu
    irgendeinem Zweck w�nschte.�

    �Zu welchem Zwecke?� fragte Rose.

    �Als ich horchte, um es zu erlauschen, erblickte er meinen Schatten an
    der Wand,� fuhr das M�dchen fort, �und es gibt au�er mir nicht sehr
    viele Menschen, die, um der Entdeckung zu entgehen, zeitig genug sich
    aus dem Hause gefunden h�tten. Mir gelang es indes, und ich sah ihn
    erst am gestrigen Abende wieder.�

    �Und was trug sich da zu?�

    �Ich will es Ihnen sagen, Lady. Er kam gestern wieder zu Fagin. Sie
    gingen wieder die Treppe hinauf; ich versteckte mich und h�llte mich so
    ein, da� mich mein Schatten nicht verraten konnte, und horchte abermals
    an der T�r. Die ersten Worte, die ich Monks sagen h�rte, waren diese:
    >So liegen denn die einzigen Beweise, da� der Oliver der Knabe ist, auf
    dem Grunde des Stromes, und die alte Hexe, die sie von seiner Mutter
    erhielt, verfault in ihrem Sarge.< Sie lachten und sprachen von der
    gl�cklichen Ausf�hrung des Streichs, und Monks, der noch weiter von
    dem Knaben sprach und sehr ingrimmig wurde, sagte: obwohl er des jungen
    Teufels Geld jetzt sicher genug h�tte, so w�rde er es doch lieber auf
    andere Art erlangt haben; denn welch eine Lust w�rde es sein, das
    prahlerische Testament des Vaters dadurch �ber den Haufen zu werfen,
    da� man den Knaben durch alle Gef�ngnisse der Hauptstadt hetzte und ihn
    dann wegen eines todesw�rdigen Verbrechens vor Gericht z�ge, was Fagin
    leicht w�rde veranstalten k�nnen, nachdem er ihn obendrein mit gro�em
    Vorteile benutzt haben w�rde.�

    �Was ist das alles?� rief Rose entsetzt aus.

    �Die Wahrheit, Lady, obwohl es von meinen Lippen kommt�, versetzte
    das M�dchen. �Dann sagte er unter Verw�nschungen, die f�r mein Ohr
    gew�hnlich genug sind, den Ihrigen aber fremd und schauerlich klingen
    m��ten, er w�rde es tun, wenn er seinen Ha� ohne Gefahr f�r seinen
    eigenen Hals dadurch befriedigen k�nnte, da� er dem Knaben das
    Leben n�hme; es d�rfte aber zu gef�hrlich sein; er w�rde ihm jedoch
    �berall im Leben auflauern und k�nnte, wenn er sich die Geburt und
    Lebensgeschichte des Knaben zunutze machte, ihm dennoch Schaden genug
    zuf�gen. >Kurzum, Fagin,< sagte er, >Jude, der du bist, du hast noch
    nie Fallstricke gelegt, wie ich sie zum Verderben meines jungen Bruders
    legen werde.<�

    �Sein Bruder!� rief Rose best�rzt.

    �Das waren seine Worte�, sagte Nancy, sich besorgt umschauend, wie sie
    es fast unabl�ssig getan hatte, denn Sikes' finstere Gestalt schwebte
    ihr best�ndig vor der Seele.

    �Und mehr noch. Indem er von Ihnen und der anderen Dame sprach, �u�erte
    er, der Himmel oder der Teufel m�sse wider ihn gewesen sein, als Oliver
    in Ihre H�nde geraten sei, und sagte mit Hohngel�chter: darin l�ge
    ebenfalls einiger Trost. Denn wieviel tausend und hunderttausend Pfund
    w�rden Sie nicht geben, wenn Sie sie h�tten, zu erfahren, wer Ihr
    zweibeiniger Scho�hund w�re.�

    �Sie wollen doch nicht sagen, da� das alles ernstlich gemeint war�,
    sagte Rose erblassend.

    �Wenn jemals ein Mensch im Ernst gesprochen, so tat ich es in diesen
    Augenblicken�, erwiderte das M�dchen, traurig den Kopf sch�ttelnd; �und
    auch er pflegt nicht zu scherzen, wenn sein Ha� in ihm lebendig ist.
    Ich kenne viele, die noch Schlimmeres �ben, aber ich w�rde sie alle
    lieber zehnmal als jenen Monks ein einziges Mal dar�ber sprechen h�ren.
    Doch es wird sp�t, und ich mu� nach Hause zur�ckkehren, um ja nicht
    den Verdacht aufkommen zu lassen, da� ich zu einem solchen Zweck hier
    gewesen w�re. Ich mu� nach Hause zur�ckeilen.�

    �Doch, was kann ich tun?� fragte Rose. �Welchen Nutzen kann ich ohne
    Sie aus Ihrer Mitteilung ziehen? Zur�ckkehren wollen Sie! Wie k�nnen
    Sie zu Genossen zur�ckzukehren w�nschen, die Sie mit so schrecklichen
    Farben schildern? Wenn Sie Ihre Aussage in Gegenwart eines Herrn,
    welchen ich augenblicklich herbeirufen kann, wiederholen wollen, so
    k�nnen Sie binnen einer halben Stunde an einen sicheren Ort gebracht
    werden.�

    �Ich w�nsche aber zur�ckzukehren�, versetzte das M�dchen. �Ich mu�
    zur�ckkehren, weil -- ach, wie kann ich mit einer unschuldigen Dame,
    wie Sie sind, �ber solche Dinge reden? -- weil unter den M�nnern, von
    welchen ich Ihnen erz�hlt habe, sich einer befindet, der Schrecklichste
    von allen, den ich nicht zu verlassen vermag; nein -- und wenn ich auch
    dadurch von dem ruchlosen, f�rchterlichen Leben erl�st werden k�nnte,
    das ich jetzt f�hre!�

    �Da� Sie zugunsten des teuren Knaben sich schon einmal bem�ht haben;
    da� Sie unter so gro�er Gefahr hierher gekommen sind, um das, was Sie
    geh�rt, mir zu enth�llen; Ihre Mienen, die mich von der Wahrheit Ihrer
    Angaben �berzeugen; Ihre offenbare Reue und Ihr Schamgef�hl: alles
    berechtigt mich dazu, zu glauben, da� Sie wieder auf den rechten Weg
    gebracht werden k�nnen. Oh�, fuhr die tiefbewegte Rose Maylie, die
    H�nde faltend, w�hrend Tr�nen �ber ihre Wangen hinabliefen, fort,
    �h�ren Sie auf das Flehen einer Angeh�rigen Ihres eigenen Geschlechts,
    der ersten -- gewi� der ersten, die jemals mit der Stimme des Mitleids
    und der Bangigkeit um Ihr Seelenheil zu Ihnen geredet hat. H�ren Sie
    auf meine Worte und lassen Sie sich durch mich zu einem besseren Dasein
    erretten!�

    �Lady,� versetzte das M�dchen, auf die Knie sinkend, �teure,
    engelgleiche Lady, ja, Sie sind die erste, die mich jemals durch Worte,
    wie diese sind, beseligt hat, und h�tte ich sie vor Jahren vernommen,
    so h�tten sie mich einem s�ndhaften und leidvollen Leben entrei�en
    k�nnen; doch es ist zu sp�t -- zu sp�t.�

    �Zur Reue und Bu�e ist es niemals zu sp�t�, entgegnete Rose.

    �Es ist dennoch zu sp�t!� rief Nancy in einem Tone aus, der ihre ganze
    Seelenqual verriet. �Ich kann ihn jetzt nicht mehr verlassen -- ich
    verm�chte es nicht, seinen Tod herbeizuf�hren.�

    �Und weshalb sollten Sie es?� fragte Rose.

    �Nichts k�nnte ihn retten�, jammerte das M�dchen. �Wenn ich anderen
    erz�hlte, was ich Ihnen offenbart habe, und dadurch seine Verhaftung
    veranla�te, er m��te ohne Rettung sterben. Er ist der verwegenste von
    allen, und hat so entsetzliche Dinge begangen!�

    �Ist es m�glich,� rief Rose, �da� Sie einem solchen Menschen zuliebe
    jeder Hoffnung auf die Zukunft und der Gewi�heit der Rettung f�r die
    Gegenwart entsagen k�nnen? Es ist Wahnsinn.�

    �Ich wei� nicht, was es ist,� versetzte das M�dchen, �ich wei� nur,
    da� es so ist, und nicht allein bei mir, sondern bei Hunderten, die
    ebenso schlecht und elend sind, wie ich es bin. Ich mu� zur�ck. Ob es
    der Zorn Gottes ist, wegen des vielen B�sen, das ich begangen habe,
    wei� ich nicht; aber ich f�hle mich trotz aller Leiden und aller harten
    Behandlung unwiderstehlich zu ihm hingezogen, was, glaub' ich, auch
    dann der Fall sein w�rde, wenn ich �berzeugt w�re, da� ich noch durch
    seine Hand sterben m��te.�

    �Was soll ich tun?� fragte Rose. �Ich m��te Sie eigentlich nicht
    fortlassen.�

    �Ja, ja, Lady, Sie werden es�, entgegnete das M�dchen. �Sie werden mein
    Fortgehen nicht hindern, weil ich in Ihre G�te Vertrauen gesetzt und
    Ihnen, wie ich es h�tte tun k�nnen, kein Versprechen abgerungen habe.�

    �Wozu n�tzt denn aber Ihre Mitteilung?� beharrte Rose. �Dies Geheimnis
    mu� enth�llt werden; welcher Vorteil kann sonst f�r Oliver, dem zu
    dienen Ihnen so sehr am Herzen liegt, daraus erwachsen, da� Sie es mir
    anvertraut haben?�

    �Sie werden sicher irgendeinen wohlwollenden Herrn kennen, dem Sie es
    mitteilen k�nnen, und der Ihnen Rat erteilen wird�, erwiderte Nancy.

    �Doch, wo finde ich Sie, wenn ich Ihrer bed�rfen sollte?� fragte Rose.
    �Ich will nicht fragen, wo jene f�rchterlichen Menschen wohnen, allein,
    wo wird man Sie an irgendeinem zu bestimmenden Tage wiedersehen k�nnen?�

    �Versprechen Sie mir, da� mein Geheimnis auf das strengste bewahrt
    werden soll, und da� Sie allein oder doch nur mit dem Manne kommen,
    dem Sie es anvertrauen wollen, und da� man mir weder auflauere noch
    nachfolge?�

    �Ich verspreche es feierlichst�, erwiderte Rose.

    �Wohlan, so will ich jeden Sonntag von elf bis zw�lf Uhr abends, wenn
    ich am Leben bleibe, auf der Londoner Br�cke auf und nieder gehen�,
    verhie� Nancy unbedenklich.

    �Warten Sie noch einen Augenblick�, sagte Rose, Nancy, die schon nach
    der T�r eilte, zur�ckhaltend. �Erw�gen Sie noch einmal Ihre Lage und
    die Gelegenheit, die Ihnen geboten wird, sich derselben zu entrei�en.
    Sie haben nicht allein als freiwillige �berbringerin einer so wichtigen
    Kunde, sondern auch als eine fast unwiederbringlich Verlorene Anspr�che
    auf meinen Beistand. Wollen Sie in der Tat zu der R�uberbande und dem
    schrecklichen Manne zur�ckkehren, da doch ein einziges Wort Sie retten
    kann? Was f�r ein Zauber ist es, der Sie unwiderstehlich zur�ckzuziehen
    und der Gottlosigkeit und dem Elend preiszugeben vermag? Ach, befindet
    sich denn in Ihrem Herzen keine Saite, die ich zu ber�hren verm�chte
    -- regt sich kein Gef�hl in ihm, das gegen diese Verblendung ank�mpfen
    k�nnte?�

    �Wenn Damen, so jung, so freundlich und sch�n, wie Sie sind, ihre
    Herzen verschenken,� versetzte das M�dchen mit fester Stimme, �so
    macht die Liebe sie zu allem f�hig -- selbst Ihresgleichen, die Sie
    eine Heimat, Angeh�rige, Freunde, zahlreiche Bewunderer, alles haben,
    was Ihr Herz ausf�llen kann. Wenn Frauen wie ich, die wir kein Dach
    als den Sargdeckel, in Krankheit und Tod keinen Beistand als die
    Krankenw�rterin des Hospitals haben, einem Manne unser angefaultes
    Herz hingeben und ihn die Stelle ausf�llen lassen, die einst von den
    Eltern, der Heimat und den Freunden ausgef�llt wurde, oder die unser
    ganzes elendes Leben hindurch eine leere und w�ste St�tte gewesen ist:
    wer kann hoffen uns zu heilen? Bemitleiden Sie uns, Lady -- bemitleiden
    Sie uns darum, da� uns nur ein weibliches Gef�hl geblieben ist, und da�
    dieses Gef�hl, durch die schwere Ahndung des Himmels, statt unser Trost
    und Stolz zu sein, zu einem Fluche und die Quelle neuer Leiden und
    Mi�handlungen wird.�

    �Sie werden doch eine Kleinigkeit von mir annehmen,� sagte Rose nach
    einer Pause, �die Sie in den Stand setzen wird, ohne Schande zu leben
    -- wenigstens bis wir uns wiedersehen?�

    �Keinen Heller�, erwiderte das M�dchen, mit der Hand abwehrend.

    �Verschlie�en Sie Ihr Herz doch nicht gegen meine Anerbietungen, Ihnen
    Beistand zu leisten�, sagte Rose, ihr n�her tretend. �Gewi�, ich
    w�nsche Ihnen n�tzlich zu sein.�

    �Sie w�rden mir am n�tzlichsten sein, Lady, wenn Sie mir mit einem
    Male das Leben nehmen k�nnten�, versetzte Nancy h�nderingend; �denn
    der Gedanke an das, was ich bin, hat mir in dieser kurzen Stunde ein
    schwereres Herzweh verursacht, als ich jemals empfunden habe, und es
    w�rde ein Gewinn sein, nicht in der H�lle zu sterben, in der ich gelebt
    habe. Gottes Segen �ber Sie, s��e Lady, und m�ge der Himmel ebensoviel
    Gl�ck auf Ihr Haupt herabsenden, wie ich auf das meine Schande geladen
    habe!�

    Mit diesen Worten und unter lautem Schluchzen verlie� die
    Bejammernswerte das Zimmer, w�hrend Rose, durch die eben beendete
    Unterredung, die mehr einem fl�chtigen Traume als der Wirklichkeit
    �hnlich sah, fast �berw�ltigt auf einen Stuhl niedersank und ihre
    verworrenen Gedanken zu sammeln suchte.




    41. Kapitel.

        Welches neue Entdeckungen enth�lt und zeigt, da� �berraschungen,
        gleich Ungl�cksf�llen, selten allein kommen.


    Roses Lage war in der Tat nicht wenig schwierig, denn w�hrend sie das
    lebhafte Verlangen empfand, das geheimnisvolle Dunkel zu durchdringen,
    das Olivers Geschichte umh�llte, konnte sie doch nicht umhin, das
    Vertrauen zu ehren, welches die Ungl�ckliche, mit der sie soeben
    gesprochen, in sie, als ein junges, argloses M�dchen, gesetzt hatte.
    Die Worte und das ganze Wesen derselben hatten Rose tief ger�hrt, und
    ihrer Zuneigung f�r ihren jugendlichen Sch�tzling gesellte sich der
    ebenso hei�e Wunsch hinzu, das Gef�hl der Reue in der Verlorenen zu
    erwecken und ihr neue Lebenshoffnung einzufl��en.

    Mrs. Maylie hatte beabsichtigt, nur drei Tage in London zu verweilen
    und dann auf einige Wochen nach einem entfernten Ort an der Seek�ste
    abzureisen. Es war Mitternacht zwischen dem ersten und zweiten Tage.
    F�r welche Schritte konnte sich Rose binnen achtundvierzig Stunden
    entscheiden? und wie konnte sie die Reise aufzuschieben suchen, ohne
    Vermutungen zu wecken, da� sich etwas Besonderes ereignet h�tte?

    Mr. Losberne weilte im Hause und beabsichtigte, auch noch die
    beiden folgenden Tage zu bleiben; allein Rose kannte die ungest�me
    Lebhaftigkeit des Ehrenmannes zu wohl und dachte sich im voraus zu
    deutlich den Zorn, welchen er im ersten Ausbruch seiner Entr�stung
    auf das Werkzeug der zweiten Entf�hrung Olivers werfen w�rde, um es
    zu unternehmen, ihm das Geheimnis, solange ihre Gegenvorstellungen
    zugunsten Nancys von keiner Seite unterst�tzt wurden, anzuvertrauen.
    Dies waren die Gr�nde, welche Rose zu dem Entschlusse bewogen, ihre
    Tante nur mit der gr��ten Vorsicht von der Sache in Kenntnis zu setzen,
    da dieselbe, wie sie voraussah, nicht verfehlen w�rde, sich mit dem
    w�rdigen Doktor �ber die Angelegenheit zu beraten. Aus denselben
    Gr�nden war nicht daran zu denken, sich an einen Rechtskundigen zu
    wenden, selbst wenn sie gewu�t, wie sie sich dabei zu benehmen h�tte.
    Einmal stieg der Gedanke in ihr auf, Harrys Beistand in Anspruch zu
    nehmen; allein er weckte die Erinnerung an ihr letztes Scheiden von ihm
    wieder auf, und es erschien ihr unw�rdig, ihn wieder in ihre N�he zu
    ziehen, da es ihm -- und bei dieser Vorstellung traten ihr die Tr�nen
    in die Augen -- jetzt vielleicht gelungen war, sie zu vergessen und auf
    eine andere Art gl�cklich zu sein.

    Durch diese wechselnden Betrachtungen aufgeregt und sich bald f�r
    diese, bald f�r jene Ma�regel entscheidend, bald alle verwerfend,
    brachte Rose die Nacht in schlafloser Bangigkeit hin und fa�te,
    nachdem sie am folgenden Tage die Sache abermals �berlegt hatte, den
    verzweifelten Entschlu�, trotz aller Bedenken Harry Maylie zu Rate zu
    ziehen.

    �Wenn es ihm auch peinlich sein wird, hierher zur�ckzukehren, ach! wie
    peinlich wird es f�r mich sein!� dachte sie sinnend. �Doch vielleicht
    kommt er nicht; er wird vielleicht schriftlich antworten oder auch
    kommen und mich �ngstlich zu meiden suchen -- wie damals, als er
    fortreiste. Ich hatte es nicht erwartet; doch es war f�r uns beide
    besser -- viel besser�; und Rose legte hier die Feder nieder und wandte
    sich hinweg, gleichsam, als ob sie zu vermeiden w�nschte, da� auch
    nur das Papier, das ihre Botschaft ausrichten sollte, ein Zeuge ihrer
    Tr�nen w�re.

    Sie hatte die Feder schon zwanzigmal wieder ergriffen und sie
    ebensooft wieder zur Seite gelegt und die Fassung der allerersten
    Zeile ihres Schreibens hin und her �berlegt, ohne auch nur eine Silbe
    niedergeschrieben zu haben, als Oliver, der mit Mr. Giles von einer
    Wanderung durch die Stadt zur�ckgekehrt war, in atemloser Hast und
    lebhafter Unruhe in das Zimmer trat, wie wenn ein neues Ungl�ck zu
    f�rchten w�re.

    �Oliver, warum siehst du so erschreckt aus?� fragte Rose, ihm
    entgegentretend. �Rede, mein Kind!�

    �Ich kann kaum; mir ist es, als ob ich ersticken m��te�, erwiderte
    der Knabe. �Ach! da� ich ihn doch noch gesehen habe, und da� Sie sich
    �berzeugen werden, da� ich Ihnen die reine Wahrheit erz�hlt habe!�

    �Ich habe nie daran gezweifelt, da� du die Wahrheit gesprochen hast,
    mein Liebling�, versetzte Rose bes�nftigend. �Doch was bedeutet dies
    alles -- von wem ist die Rede?�

    �Ich habe den Herrn gesehen,� erwiderte der Knabe, �den Herrn, der so
    g�tig gegen mich war -- Mr. Brownlow, von dem wir so oft gesprochen
    haben.�

    �Wo?� fragte Rose.

    �Er stieg eben aus einem Wagen und ging in ein Haus�, erwiderte Oliver,
    indem Freudentr�nen aus seinen Augen hervorst�rzten. �Ich redete ihn
    nicht an -- ich konnte nicht, denn er sah mich nicht, und ich zitterte
    so, da� ich nicht imstande war, zu ihm zu gehen. Aber Giles erkundigte
    sich, ob er in dem Hause wohnte, und man sagte ja. Sehen Sie,� fuhr er
    fort, ein St�ck Papier entfaltend, �hier steht es; da wohnt er -- ich
    will sogleich hingehen. O Gott! ich werde mich nicht fassen k�nnen,
    wenn ich ihn sehe und seine Stimme wieder h�re!�

    Rose Maylie hatte unter diesen und noch vielen �hnlichen Ausrufen
    der Freude des Knaben gro�e M�he, Mr. Brownlows Adresse zu lesen,
    >Craven Street, Strand<; sie beschlo� indes nach einiger Zeit, Olivers
    Entdeckung ohne S�umen zu benutzen.

    �Schnell!� sagte sie, �gib Befehl, einen Mietwagen kommen zu lassen,
    und halte dich bereit, mich zu begleiten. Ich werde dich, ohne einen
    Augenblick zu verlieren, hinbringen und will nur erst meiner Tante
    sagen, da� wir auf eine Stunde ausfahren wollen; ich werde ebenso rasch
    fertig sein wie du selbst.�

    Es bedurfte bei Oliver keiner Mahnung zur Eile, und in weniger als f�nf
    Minuten befanden sie sich auf dem Wege nach der bezeichneten Stra�e.
    Als sie angelangt waren, lie� Rose Oliver unter dem Vorwande, den alten
    Herrn auf sein Erscheinen vorzubereiten, allein im Wagen, stieg aus
    und schickte durch den Diener ihre Karte mit der Bitte hinauf, Mr.
    Brownlow in sehr dringenden Angelegenheiten sprechen zu d�rfen. Der
    Diener kehrte bald wieder zur�ck, um sie zu ersuchen, hinaufzukommen.
    Sie folgte ihm in eins der oberen Zimmer, wo sie einen �ltlichen, in
    einem dunkelgr�nen Rocke sich pr�sentierenden Herrn, in dessen Mienen
    unverkennbare Herzensg�te sich ausdr�ckte, fand. Nicht weit von ihm
    erblickte sie einen zweiten alten Herrn in Nankingbeinkleidern und
    Gamaschen, der nicht besonders wohlwollend aussah und dasa�, die H�nde
    auf den Knauf eines schweren Spazierstocks gest�tzt und das Kinn auf
    demselben ruhend lassend.

    �Ah!� sagte der Herr im gr�nen Rocke, eilfertig und mit Zuvorkommenheit
    aufspringend, �ah, entschuldigen Sie, mein gn�diges Fr�ulein -- ich
    glaubte, es w�re irgendeine zudringliche Person, die -- Sie werden mich
    g�tigst entschuldigen. Bitte, nehmen Sie Platz!�

    �Mr. Brownlow, wenn ich nicht irre, Sir?� sagte Rose, nachdem sie auf
    den andern Herrn einen Blick geworfen hatte.

    �So ist mein Name, ja�, erwiderte der alte Herr. �Dies ist mein Freund,
    Mr. Grimwig. Grimwig, Sie haben wohl die Gef�lligkeit und verlassen uns
    auf einige Minuten.�

    �Ich glaube nicht, da� es notwendig sein wird, den Herrn zu bem�hen�,
    bemerkte Rose. �Wenn ich nicht irre, so ist ihm die Angelegenheit, in
    welcher ich Sie zu sprechen w�nsche, nicht fremd.�

    Brownlow gab seine Einwilligung durch eine leichte Kopfneigung zu
    erkennen, und Grimwig, der eine sehr steife Verbeugung gemacht hatte
    und aufgestanden war, machte eine zweite sehr steife Verbeugung und
    nahm wieder Platz.

    �Was ich Ihnen mitzuteilen habe, wird Sie ohne Zweifel sehr
    �berraschen�, begann Rose etwas verlegen. �Sie erwiesen einst einem
    mir sehr teuern jungen Freunde viel Wohlwollen und G�te, und ich bin
    �berzeugt, da� es Sie freuen wird, wieder von ihm zu h�ren.�

    �Einem jungen Freunde!� sagte Mr. Brownlow. �Darf ich seinen Namen
    wissen?�

    �Oliver Twist!� erwiderte Rose.

    Kaum waren die Worte ihrem Munde entflohen, als Grimwig, der sich
    gestellt hatte, als ob ihn der Inhalt eines auf dem Tisch liegenden
    gro�en Buches lebhaft interessierte, dasselbe mit gro�em Ger�usche
    zuschlug, sich zur�cklehnte, wobei sein Antlitz den Ausdruck des
    �u�ersten Erstaunens annahm, und lange mit gro�en, stieren Augen dasa�,
    worauf er, als ob er sich sch�mte, so viel innere Bewegung an den
    Tag gelegt zu haben, sich in seine vorige Stellung gleichsam wieder
    zur�ckschnellte und, indem er gerade vor sich hinstarrte, einen langen,
    pfeifenden Ton erschallen lie�, der nicht in der leeren Luft, sondern
    in den innersten H�hlen seines Magens zu ersterben schien.

    Mr. Brownlow war nicht weniger erstaunt, wiewohl sein Erstaunen sich
    auf eine weit minder seltsame Art kundgab. Er r�ckte seinen Stuhl n�her
    zu Rose heran und sagte: �Erzeigen Sie mir den Gefallen, mein liebes,
    gn�diges Fr�ulein, die G�te und das Wohlwollen, von welchem Sie reden,
    und wovon, Sie ausgenommen, niemand wei�, g�nzlich au�er Frage zu
    lassen; und wenn Sie irgend Beweise herbeizubringen verm�gen, welche
    geeignet sind, mir die �ble Meinung zu benehmen, die ich vormals von
    dem genannten ungl�cklichen Kinde zu hegen mich bewogen fand, so teilen
    Sie sie mir mit -- ich bitte dringend darum.�

    �Ein b�ser Bube -- ich will meinen Kopf aufessen, wenn er es nicht
    ist�, brummte Grimwig in sich hinein wie ein Bauchredner und ohne einen
    Gesichtsmuskel in Bewegung zu setzen.

    �Der Knabe besitzt einen reinen Sinn und ein warmes Herz�, sagte Rose,
    vor Unmut err�tend; �und die Allmacht, der es gefallen, ihm Pr�fungen,
    die �ber seine Jahre hinausgingen, aufzuerlegen, hat in seiner Brust
    Gef�hle und Gesinnungen keimen lassen, welche unz�hligen Ehre machen
    w�rden, die seine Jahre sechsfach z�hlen.�

    �Ich bin erst einundsechzig,� bemerkte Grimwig mit derselben starren
    Unbeweglichkeit, �und da es mit dem Teufel zugehen m��te, wenn dieser
    Oliver nicht wenigstens zw�lf Jahr alt ist, so sehe ich das Zutreffende
    der Bemerkung nicht ein.�

    �Achten Sie nicht auf meinen Freund, Mi� Maylie�, sagte Brownlow; �er
    meint es doch nicht so.�

    �Das tut er allerdings�, brummte Grimwig vor sich hin.

    �Nein, er tut es nicht�, beharrte Brownlow, der offenbar immer
    erz�rnter wurde.

    �Er will seinen Kopf aufessen, wenn er es nicht tut�, beteuerte Grimwig.

    �Er verdiente, ihn zu verlieren, wenn er es t�te�, entgegnete Brownlow.

    �Und er m�chte denjenigen sehen, der es zu versuchen wagte, ihm den
    Kopf zu nehmen�, erwiderte Grimwig, seinen Stock mit Heftigkeit gegen
    den Fu�boden sto�end.

    Nachdem die alten Herren soweit gediehen waren, nahmen beide eine
    Prise Schnupftabak und dr�ckten darauf, gem�� ihrer unab�nderlichen
    Gewohnheit, einander die H�nde.

    �Und nun, Mi� Maylie,� begann Brownlow, sich wieder zu Rose wendend,
    �lassen Sie uns zu dem Gegenstande zur�ckkehren, an welchem Ihre
    Menschenliebe einen so gro�en Anteil nimmt. Darf ich wissen, welche
    Kunde Sie von dem armen Knaben besitzen? Erlauben Sie mir, Ihnen vorher
    mitzuteilen, da� ich, um ihn ausfindig zu machen, alle mir zu Gebote
    stehenden Mittel ersch�pft habe, und da� seit meiner Reise au�er Landes
    meine erste Meinung, da� er mich belogen und durch seine ehemaligen
    Genossen beredet gewesen, mich zu bestehlen, bedeutend ersch�ttert
    worden ist.�

    Rose, welcher diese Rede Zeit gelassen hatte, ihre Gedanken zu sammeln,
    berichtete nun Brownlow alles, was sich mit Oliver zugetragen, seit er
    das Haus desselben verlassen, und verschwieg ihm einstweilen nur Nancys
    unter vier Augen ihm anzuvertrauende Mitteilungen. Sie schlo� mit
    der Versicherung, da� der einzige Kummer, den der Knabe seit einigen
    Monaten empfunden, dem Umstande zuzuschreiben sei, da� er seinen
    ehemaligen Wohlt�ter und Freund nirgends habe finden k�nnen.

    �Gott sei Dank!� rief der alte Herr. �Diese Nachricht macht mich
    gl�cklich, sehr gl�cklich. Doch Sie haben mir nicht gesagt, wo er sich
    gegenw�rtig befindet, Mi� Maylie. Verzeihen Sie mir -- doch weshalb
    haben Sie ihn nicht mitgebracht?�

    �Er wartet im Wagen vor der T�r�, erwiderte Rose.

    �Vor meiner T�r?� rief der alte Herr freudig �berrascht aus, eilte,
    ohne ein Wort zu sagen, hinaus, die Treppe hinunter, sprang auf den
    Wagentritt und in den Wagen hinein.

    Sobald er fort war, hob Grimwig den Kopf empor, balancierte seinen
    Stuhl auf einem Hinterbeine und beschrieb, ohne aufzustehen und mit
    Hilfe seines Stockes und des Tisches, drei ganze Kreise. Nachdem er die
    Evolution gl�cklich ausgef�hrt hatte, sprang er auf und humpelte nach
    besten Kr�ften zum wenigsten ein dutzendmal im Zimmer auf und ab, blieb
    pl�tzlich vor Rose stehen und k��te sie ohne alle weitere Einleitung.

    �Pst!� sagte er, als Rose, �ber dieses ungew�hnliche Verfahren ein
    wenig erschreckt, aufstand; �seien Sie ohne Furcht. Ich bin alt genug,
    um Ihr Gro�vater zu sein. Sie sind ein wackeres, ein sehr gutes
    M�dchen -- ich habe Sie lieb. Da kommen sie!�

    Bei diesen Worten warf er sich mit einer geschickten Wendung auf seinen
    Sitz, und in demselben Augenblick trat Brownlow mit Oliver herein, den
    Grimwig sehr gn�dig begr��te. Ach, wenn die Freude dieses Augenblicks
    Roses einzige Belohnung gewesen w�re f�r alle ihre Sorge und Angst um
    Oliver, sie w�rde sich hinl�nglich belohnt gef�hlt haben.

    �Es ist aber noch jemand, den wir nicht vergessen d�rfen�, sagte
    Brownlow, nach der Klingelschnur greifend.

    �Sage Mrs. Bedwin, sie m�chte einmal heraufkommen�, befahl er dem
    hereintretenden Diener.

    Die bejahrte Haush�lterin erschien sogleich, machte ihren Knicks und
    blieb, des Befehls des Herrn gew�rtig, an der T�r stehen.

    �Mein Gott, Sie werden ja alle Tage blinder�, sagte Brownlow ein wenig
    verdrie�lich.

    �Mag wohl sein, Sir�, erwiderte die gute Alte. �In meinen Jahren
    pflegen die Augen nicht sch�rfer zu werden, Sir.�

    �Das h�tte ich Ihnen auch sagen k�nnen�, entgegnete Brownlow. �Doch
    setzen Sie Ihre Brille auf und sehen Sie zu, ob Sie nicht selbst
    entdecken k�nnen, weshalb ich Sie habe heraufkommen lassen.�

    Die alte Frau begann sogleich in ihren Taschen zu w�hlen; aber Olivers
    Geduld hielt die Probe nicht aus, er �berlie� sich dem Drange seiner
    Gef�hle und warf sich in ihre Arme.

    �Gott sei mir gn�dig! -- es ist mein unschuldiger Knabe!� rief sie aus,
    indem sie ihn z�rtlich in die Arme dr�ckte.

    �Meine liebe alte Pflegemutter!� rief Oliver.

    �Gott, ich wu�te es wohl, da� er zur�ckkehren w�rde! Wie gesund und
    bl�hend er aussieht, und er ist obendrein wie 'nes Edelmannes Sohn
    gekleidet! Wo bist du so lange, so lange gewesen! Ach! es ist dasselbe
    s��e Gesichtchen, aber nicht so bla�; dasselbe sanfte Auge, aber nicht
    so tr�be. Sie sind mir gar nicht aus dem Sinn gekommen und ihr stilles
    L�cheln auch nicht; ich habe sie Tag f�r Tag neben meinen lieben
    Kindern gesehen, die, seit ich ein gl�ckliches junges Weib war, tot und
    dahingegangen sind.�

    Sich so ihrer Redseligkeit �berlassend und Oliver bald von sich
    haltend, um ihn genauer ansehen zu k�nnen, und ihn bald z�rtlich an die
    Brust dr�ckend und ihm die Locken aus dem Gesichte streichend, weinte
    und lachte die gute alte Seele in einem Atem.

    Brownlow �berlie� beide dem Austausch ihrer Gef�hle und f�hrte Rose
    in ein anderes Zimmer, wo sie ihm einen ausf�hrlichen Bericht �ber
    die Unterredung mit Nancy erstattete, die ihn nicht wenig �berraschte
    und in Verwirrung und Unruhe setzte. Rose teilte ihm auch ihre Gr�nde
    mit, weshalb sie nicht ihren Freund Losberne zun�chst zum Vertrauten
    gemacht h�tte. Der alte Herr �u�erte, sie habe daran sehr klug getan,
    und erkl�rte sich bereit, mit dem w�rdigen Doktor selbst in Beratung zu
    treten. Um ihm hierzu eine baldige Gelegenheit zu verschaffen, wurde
    verabredet, da� er noch an demselben Abend in der Villa vorsprechen
    und da� mittlerweile Mrs. Maylie von allem, was vorgefallen war,
    vorsichtig in Kenntnis gesetzt werden sollte. Sobald diese vorl�ufigen
    Bestimmungen getroffen waren, kehrten Rose und Oliver wieder nach Hause
    zur�ck.

    Rose hatte das Ma� der Entr�stung des trefflichen Doktors keineswegs
    �bersch�tzt; denn kaum war ihm Nancys Erz�hlung mitgeteilt worden,
    als er seinen Zorn in einem Strome von Verw�nschungen und Drohungen
    ergo�, sie zum ersten Schlachtopfer des vereinten Scharfsinnes
    der Herrn Blathers und Duff zu machen gelobte und sogar den Hut
    aufsetzte, in der Absicht, fortzueilen und den Beistand der genannten
    Ehrenm�nner in Anspruch zu nehmen. Und er w�rde im ersten Losst�rmen
    sein Vorhaben, ohne die Folgen des allergeringsten Nachdenkens zu
    w�rdigen, ohne Zweifel ausgef�hrt haben, wenn er nicht zur�ckgehalten
    worden w�re, teils durch das ebenso gro�e Ungest�m Brownlows, der
    selbst ein reizbares Temperament besa�, und teils durch die Gr�nde und
    Gegenvorstellungen, die man f�r die zweckdienlichsten erachtete, ihn
    von seinem unbesonnenen Verfahren zur�ckzubringen.

    �Was ist aber zum Geier zu tun?� sagte der hitzige Doktor, als sie in
    das Zimmer zu den beiden Damen getreten waren. �Wir sollen doch nicht
    all das m�nnliche und weibliche Gesindel unseres Dankes versichern und
    es bitten, hundert oder ein paar hundert Pfund als ein geringes Zeichen
    unserer Achtung und als einen kleinen Beweis unserer Erkenntlichkeit
    f�r ihre G�te gegen Oliver anzunehmen?�

    �Das eben nicht�, erwiderte Brownlow mit Lachen; �allein wir m�ssen
    besonnen und mit gro�er Vorsicht handeln.�

    �Besonnen und vorsichtig!� rief der Doktor aus. �Ich w�rde die Halunken
    samt und sonders zum --�

    �Es ist einerlei, zu wem Sie sie schicken w�rden�, unterbrach ihn
    Brownlow. �Doch fragen Sie sich selbst, ob wir, wir m�gen sie schicken,
    wohin wir wollen, eine Hoffnung haben, dadurch zum Ziele zu gelangen.�

    �Zu welchem Ziele?� fragte der Doktor.

    �Dem einfachen Ziele, zu erforschen, wer Olivers Eltern gewesen sind,
    und ihm seine Erbschaft wieder zuzuwenden, um welche er, sofern alle
    vorliegenden Angaben begr�ndet sind, sch�ndlich betrogen worden ist.�

    �Ah!� sagte Losberne, sich mit dem Schnupftuche K�hlung zuwehend, �das
    h�tte ich bald vergessen.�

    �Sie begreifen also�, fuhr Brownlow fort. �Was w�rden wir denn Gutes
    stiften, wenn wir, angenommen, es w�re ausf�hrbar, ohne die Sicherheit
    des armen M�dchens zu gef�hrden, die B�sewichter dem Arme der
    Gerechtigkeit �berlieferten?�

    �Das Gute, da� einige von ihnen baumelten und die �brigen deportiert
    w�rden�, meinte der Doktor.

    �Sehr wohl�, erwiderte Brownlow l�chelnd; �allein sie werden daf�r
    seinerzeit ohne Zweifel schon selber sorgen, und wenn wir ihnen
    vorgreifen, so scheint mir's, wir werden eine arge Don-Quichotterie
    begehen und unserm oder doch Olivers Interesse, was aber dasselbe ist,
    gerade zuwiderhandeln.�

    �Wieso denn?� fragte der Doktor.

    �Ist es nicht klar genug,� erwiderte Brownlow, �da� es uns �u�erst
    schwer werden wird, dem Geheimnisse auf den Grund zu kommen, wenn wir
    nicht imstande sind, Monks zum Beichten zu bringen? Das kann aber nur
    durch List geschehen und dadurch, da� wir ihn fassen, wenn er eben
    nicht von dem �brigen Gelichter umgeben ist. Denn gesetzt auch, da�
    er aufgegriffen w�rde -- wir haben keine Beweise wider ihn. Er hat
    (soviel wir wissen, oder so weit es aus den Umst�nden hervorgeht) an
    keinem Diebstahle oder Raube der Bande teilgenommen. Wenn er auch nicht
    freigesprochen werden w�rde, so ist es doch sehr unwahrscheinlich, da�
    er eine weitere Strafe erhielte als die, da� man ihn eine Zeitlang als
    Landstreicher einsperrte, und sein Mund w�rde dann hinterher f�r immer
    so fest geschlossen sein, da� wir unsere Zwecke ebensowenig erreichten,
    wie wenn er taub, stumm, blind und bl�dsinnig w�re.�

    �Ich frage Sie abermals,� sagte der Doktor heftig, �ob Sie das dem
    M�dchen gegebene Versprechen vern�nftigerweise f�r bindend halten --
    ein Versprechen, das in der besten und wohlwollendsten Absicht gegeben
    ist, aber wirklich --�

    �Ich bitte, lassen Sie den Punkt uner�rtert, mein verehrtes junges
    Fr�ulein�, sagte Brownlow, Rose zuvorkommend; �das Versprechen soll
    gehalten werden. Ich glaube nicht, da� es unseren Schritten auch
    nur im mindesten hinderlich sein wird. Doch bevor wir bestimmte
    Entschlie�ungen in betreff der zu ergreifenden Ma�regeln fassen k�nnen,
    m�ssen wir notwendig das M�dchen sehen, um von ihr zu h�ren, ob sie
    uns so oder anders dazu verhelfen will oder kann, Monks' habhaft zu
    werden, oder wenn nicht, sie wenigstens zu bewegen, uns seine Person zu
    beschreiben und uns zu sagen, wo er sich zu verstecken pflegt, oder
    wo er sonst zu finden sein mag. Das kann nun vor dem n�chsten Sonntag
    abend nicht geschehen, und heute ist Dienstag. Mein Rat ist daher, da�
    wir uns bis dahin vollkommen ruhig und die Sache selbst vor Oliver
    geheimhalten.�

    Obgleich Mr. Losberne zu dem Vorschlage, f�nf ganze Tage unt�tig
    zu sein, die sauerste Miene machte, so mu�te er doch zugeben, im
    Augenblick keinen besseren Rat zu wissen; und da sowohl Rose wie Mrs.
    Maylie auf Brownlows Seite traten, so wurde des letzteren Rat endlich
    allerseits gebilligt.

    �Ich n�hme gern den Beistand meines Freundes Grimwig in Anspruch�,
    sagte Brownlow. �Er ist ein wunderlicher Kauz, besitzt aber sehr
    viel Scharfblick und k�nnte uns von wesentlichem Nutzen sein. Er ist
    Rechtsgelehrter von Haus aus und entsagte lediglich aus Unmut dar�ber,
    da� ihm binnen zehn Jahren nur ein einziger Proze� anvertraut wurde,
    dem Advokatenstande. Sie m�gen indes selbst entscheiden, ob das eine
    Empfehlung ist oder nicht.�

    �Ich habe nichts dawider, da� Sie Ihren Freund zuziehen, wenn ich auch
    den meinigen zuziehen darf,� sagte Losberne und erwiderte auf Brownlows
    Frage, wer derselbe w�re: �Der Sohn Mrs. Maylies und Mi� Roses -- sehr
    alter Freund.�

    Roses Wangen wurden purpurn; sie machte jedoch keine h�rbare Einwendung
    gegen den Vorschlag (vielleicht weil sie erkannte, da� sie doch
    jedenfalls in einer hoffnungslosen Minorit�t bleiben w�rde), und Harry
    Maylie und Grimwig wurden daher zu Mitgliedern des Komitees ernannt.

    �Wir bleiben nat�rlich so lange in der Stadt,� sagte Mrs. Maylie, �wie
    noch die geringste Aussicht vorhanden ist, unsere Nachforschungen
    mit Erfolg fortzusetzen. Ich werde bei einer uns alle so sehr
    interessierenden Sache weder M�he noch Kosten sparen und gern
    hierbleiben, und wenn es sein mu�, zw�lf Monate, solange Sie mir sagen
    k�nnen, da� noch Hoffnung vorhanden ist.�

    �Gut�, versetzte Brownlow; �und da ich auf Ihren Gesichtern lese, da�
    Sie mich fragen wollen, wie es zugegangen ist, da� ich nicht zur Stelle
    war, Olivers Erz�hlung zu best�tigen, und da� ich das Land so pl�tzlich
    verlassen, so erlauben Sie mir, die Forderung zu stellen, da� mir nicht
    eher Fragen vorgelegt werden, als bis ich es f�r geeignet erachte,
    denselben durch meine Geschichte zuvorzukommen. Glauben Sie mir, meine
    Forderung hat ihren guten Grund; denn wenn ich von ihr abginge, k�nnte
    ich vielleicht Hoffnungen erwecken, welche nie verwirklicht w�rden und
    nur alle schon hinl�nglich zahlreichen Schwierigkeiten und T�uschungen
    noch vermehrten. -- Meine Herrschaften, wir sind zum Abendessen
    gerufen, und Oliver, der einsam im ansto�enden Zimmer weilt, wird am
    Ende glauben, da� wir seiner m�de geworden w�ren und einen finsteren
    Anschlag ausd�chten, ihn wieder in die Welt hinauszusto�en.�

    Mit diesen Worten reichte der alte Herr Mrs. Maylie die Hand und f�hrte
    sie in das Speisezimmer; Losberne folgte mit Rose, und die Beratung
    hatte damit ein Ende.




    42. Kapitel.

        Ein alter Bekannter von Oliver l��t entschiedene Geniespuren
        blicken und wird ein �ffentlicher Charakter in der Hauptstadt.


    Gerade an dem Abende, an welchem Nancy ihre selbstauferlegte Mission
    bei Rose Maylie erf�llte, wanderten auf der gro�en, nach Norden
    f�hrenden Heerstra�e zwei Personen nach London, denen wir einige
    Aufmerksamkeit widmen m�ssen. Die eine derselben, eine Mannsperson,
    geh�rte zu den langen, kn�chernen Gestalten, die als Knaben wie
    verk�mmerte M�nner, und wenn sie fast M�nner sind, wie zu fr�h gro�
    gewordene Knaben aussehen. Die zweite, ein Frauenzimmer, war jung, aber
    derb und kr�ftig, was sie auch sein mu�te, um unter der schweren B�rde
    auf ihrem R�cken nicht zu erliegen. Ihr Begleiter trug nur weniges und
    leichtes Gep�ck an einem Stocke �ber der Schulter und konnte daher um
    so leichter, zumal da ihm auch die L�nge seiner Beine zustatten kam,
    stets einige Schritte weit voran sein, woran er es auch ebensowenig
    fehlen lie� wie an h�ufigen Vorw�rfen, die er seiner Gef�hrtin wegen
    ihrer Langsamkeit machte. Sie hatten Highgate hinter sich, als er
    stillstand und ihr ungeduldig zurief: �Kannst du nicht geschwinder
    gehen? Was schleichst du immer so faul von weitem nach, Charlotte?�

    �'s ist 'ne schwere Tracht, das kannst du nur glauben�, erwiderte sie,
    fast atemlos herankommend.

    �Schwer? Was ist das f�r Schw�tzen -- wozu hab' ich dich?� fuhr Noah
    Claypole (denn er war es) fort und legte sein kleines B�ndel auf die
    andere Schulter. �Und nun stehst du schon wieder still? Bei dir mu�
    auch der Beste die Geduld verlieren.�

    �Ist es noch weit?� fragte Charlotte, indem ihr die Schwei�tropfen �ber
    das Gesicht herabstr�mten.

    �Noch weit? Wir sind schon so gut wie da. Sieh hin -- dort sind die
    Lichter von London.�

    �Dann sind wir wenigstens noch zwei gute Meilen davon entfernt�, sagte
    Charlotte verzweiflungsvoll.

    �Zwei Meilen oder zwanzig ist auch einerlei; steh auf und mach fort,
    oder du bekommst Fu�tritte�, warnte Noah mit vor Zorn noch mehr als
    gew�hnlich ger�teter Nase, und Charlotte stand auf und schritt wieder
    neben ihm her.

    �Wo denkst du f�r diese Nacht einzukehren, Noah?� fragte sie nach
    einiger Zeit.

    �Was wei� ich's?� antwortete Mr. Claypole, den das lange Gehen
    verdrie�lich gemacht hatte.

    �Doch in der N�he?�

    �Nein, nicht in der N�he.�

    �Warum denn nicht?�

    �Wenn ich dir sage, da� ich das will oder nicht will, so ist's genug,
    ohne da� du zu fragen brauchst, warum oder weshalb�, entgegnete Noah
    mit W�rde.

    �Ich frage ja nur -- brauchst ja nicht so b�se dar�ber zu werden.�

    �Das w�r' mir wohl ein recht kluger Streich, im ersten besten
    Wirtshause vor der Stadt einzukehren, da� Sowerberry, wenn er uns
    etwa nachsetzte, seine alte Nase hereinsteckte und uns gleich wieder
    fest h�tte und mit Handschellen zur�ckbr�chte! Nein, ich werde in
    die engsten Stra�en einlenken, die ich finden kann, und nicht eher
    haltmachen, als bis wir das entlegenste Gasthaus gefunden haben. Du
    kannst deinem Sch�pfer danken, da� ich Pfiffigkeit f�r dich mit habe;
    denn wenn wir nicht auf meinen Rat erst den entgegengesetzten Weg
    eingeschlagen h�tten, so w�rst du schon vor acht Tagen eingesperrt, und
    dir w�re recht geschehen als 'ner dummen Gans.�

    �Ich wei� es, da� ich nicht so klug bin wie du; aber wirf nur nicht
    alle Schuld auf mich allein. W�r' ich eingesperrt, w�rdest du es doch
    auch sein.�

    �Du wei�t doch wohl, da� du das Geld aus dem Ladentische nahmst?�

    �O ja, lieber Noah, aber ich nahm es f�r dich.�

    �Nahm ich's hin und trug's bei mir?�

    �Nein; du vertrautest mir und lie�est 's mich tragen, und das war doch
    gut von dir�, sagte Charlotte, ihn unter das Kinn klopfend und ihren
    Arm in den seinigen legend.

    Es verhielt sich in der Tat so; allein es war Mr. Claypoles Weise
    nicht, in irgend jemand ein blindes und t�richtes Vertrauen zu setzen,
    und wir lassen ihm nur Gerechtigkeit widerfahren, wenn wir bemerken,
    da� er Charlotten lediglich deshalb so sehr vertraut hatte, damit das
    Geld, wenn sie verfolgt w�rden, bei ihr gefunden werden m�chte. Er lie�
    sich jedoch bei dieser Gelegenheit nat�rlich auf keine Darlegung seiner
    Beweggr�nde ein, und beide wanderten im z�rtlichsten Einvernehmen
    miteinander weiter.

    Seinem vorsichtigen Plane zufolge schritt Mr. Claypole, ohne
    anzuhalten, bis nach dem Engel von Islington weiter, wo er aus dem
    beginnenden Gedr�nge der Fu�g�nger und Fuhrwerke sehr scharfsinnig
    schlo�, da� London nunmehr ernstlich anfinge. Er schaute nun einen
    Augenblick umher, welche Stra�en die belebtesten und also am meisten zu
    meidenden schienen, lenkte in St. Johns Road ein und befand sich bald
    tief in dem Gewirr obskurer und schmutziger Stra�en und Gassen zwischen
    Grays Inn Lane und Smithfield, einem Stadtteile mitten in London, der
    trotz der allgemeinen Fortschritte und ungemeiner Versch�nerungen
    abscheulich geblieben ist.

    Noah schaute fortw�hrend nach einem Gasthause aus, wie er es sich bei
    seinen Zwecken und seiner Lage w�nschenswert dachte, stand endlich vor
    dem elendesten still, das er bis dahin gesehen hatte, und erkl�rte,
    hier f�r die Nacht einkehren zu wollen.

    �Gib mir nun das B�ndel,� sagte er, es seiner Begleiterin abnehmend,
    �und sprich nicht, au�er wenn du angeredet wirst. Wie nennt sich das
    Haus? Was steht da -- d-r-e-i --?�

    �Kr�ppel�, fiel Charlotte ein.

    �Drei Kr�ppel -- ein sehr guter Name,� bemerkte Noah. �Halt dich dicht
    hinter mir -- vorw�rts!�

    Er stie� die gebrechliche T�r mit den Schultern auf, und beide gingen
    hinein. Im Schenkst�bchen war niemand, als ein junger Mensch, ein Jude,
    der in einem schmutzigen Zeitungsblatte las. Er starrte Noah und Noah
    starrte ihn an.

    �Sind dies die drei Kr�ppel?� fragte Noah.

    �So nennt sich das Haus.�

    �Wir trafen 'nen Gentleman, der uns hierher rekommandiert hat�, fuhr
    Noah, Charlotte ansto�end, fort, vielleicht, um sie aufmerksam auf
    seine List zu machen, sich Achtung zu verschaffen, oder vielleicht um
    sie zu erinnern, ihn nicht zu verraten. �Wir m�chten hier �bernachten.�

    �Ich wei� nicht, ob es geht an,� erwiderte Barney -- denn er war der
    dienende Geist dieses Hauses -- �will aber anfragen.�

    �Bringt uns unterdes in die Gaststube und gebt uns 'nen Mund voll
    kaltes Fleisch und 'nen Schluck Bier�, sagte Noah.

    Barney f�hrte die m�den Reisenden in ein Hinterzimmer, brachte ihnen
    die geforderten Erfrischungen, teilte ihnen zugleich mit, da� sie �ber
    Nacht bleiben k�nnten, und lie� das liebensw�rdige P�rchen allein. --
    Das Zimmer, in welches er sie gef�hrt hatte, befand sich unmittelbar
    hinter dem Schenkst�bchen und lag einige Fu� niedriger, so da� man aus
    jenem, wenn man von einem Diminutivfensterchen etwas hoch in der Wand
    einen Vorhang zur�ckschob, ohne bemerkt zu werden, genau sehen und
    h�ren konnte, was die G�ste darin vornahmen oder sprachen. Noah und
    Charlotte hatten sich kaum zu ihrem Imbisse niedergesetzt, als Fagin im
    Schenkst�bchen erschien, um nach einem seiner jungen Z�glinge zu fragen.

    �Pst!� sagte Barney; �es sind nebenan Fremde.�

    �Fremde?� wiederholte Fagin fl�sternd.

    �Ja -- nicht aus der Stadt, kurioses Volk; und ich m��te irren sehr,
    wenn sie nicht was w�ren f�r Euch.�

    Fagin stieg sogleich auf einen Stuhl und sah durch das kleine Fenster,
    wie Noah tapfer schmauste und Charlotte von Zeit zu Zeit hom�opathische
    Dosen zuteilte.

    �Aha!� fl�sterte Fagin, zu Barney sich umdrehend, �die Miene des
    Burschen k�nnte gefallen mir. Er w�rde uns sein k�nnen n�tzlich, denn
    er versteht's schon, zu kirren die Dirne. Sei stiller als eine Maus,
    mein Lieber, da� ich sie h�re sprechen.�

    Er schaute abermals durch das kleine Fenster, und zwar mit einem
    Gesichte, das einem alten Gespenst angeh�rt haben k�nnte.

    �Ich denke also von jetzt ab ein Gentleman zu sein,� sagte Noah,
    die Beine ausstreckend und ein Gespr�ch fortsetzend, dessen Anfang
    dem Juden entgangen war. �Nichts mehr von S�rgen und Aufwarten bei
    Herrschaften, Charlotte, sondern nunmehr wie ein Gentleman gelebt; und
    wenn du willst, sollst du 'ne Dame werden.�

    �Ei, das m�cht' ich freilich wohl, lieber Noah�, antwortete Charlotte;
    �aber es gibt nicht alle Tage Ladenkassen zu leeren und so, da� man
    nachher gut davonkommt.�

    �Hol' der Geier alle Ladenkassen!� rief Noah aus, �es gibt noch mehr
    Dinge, die geleert werden k�nnen.�

    �Was meinst du denn?� fragte Charlotte.

    �Taschen, Strickbeutel, H�user, Postkutschen, Banken�, erwiderte Mr.
    Claypole, dem der Mut wuchs, indem ihm der Porter zu Kopfe stieg.

    �Du kannst das aber nicht alles, lieber Noah�, sagte Charlotte.

    �Ich werde mich nach Genossen umsehen, die es verm�gen�, versetzte
    Noah. �Sie werden uns auf die eine oder andere Weise gebrauchen k�nnen.
    Du bist selbst soviel wie f�nfzig Weibsbilder wert; denn ich hab' nie
    eins gekannt, das so voll List und Trug steckte wie du, wenn ich dir
    freie Hand lasse.�

    �Jemine, wie du flattieren kannst!� rief Charlotte aus und dr�ckte ihm
    einen Ku� auf den h��lichen Mund.

    �La� gut sein�, sagte Noah mit gro�er W�rde, sich von ihr losmachend;
    �sei ja nicht zu z�rtlich, wenn ich b�se mit dir bin. Ich wollte, da�
    ich Hauptmann 'ner Bande w�r', h�tte sie unter der Zucht und folgte
    ihnen allerw�rts nach, ohn' da� sie's selber w��ten. Das w�r' so
    was f�r mich, wenn's guten Profit abw�rfe; und h�r', wenn's uns nur
    gl�ckte, da� uns einige Gentlemen von dieser Sorte in den Wurf k�men,
    es w�r' uns soviel wert wie unsere Zwanzigpfundnote -- besonders da wir
    eigentlich nicht wissen, wie wir sie loswerden sollen.�

    Mr. Claypole blickte bei diesen Worten mit �u�erst weiser Miene in den
    Porterkrug hinein, trank, nickte Charlotte herablassend zu und stand
    im Begriff, einen zweiten Zug zu tun, als die T�r sich auftat und die
    Erscheinung eines Unbekannten ihn unterbrach. Der Unbekannte war Mr.
    Fagin. Er hatte seine einnehmendste Miene angenommen, n�herte sich mit
    einer sehr tiefen Verbeugung, nahm an einem Tische dicht neben dem, an
    welchem das P�rchen sa�, Platz und rief dem grinsenden Barney zu, ihm
    einen Trunk zu bringen.

    �Ein angenehmer Abend, Sir, nur k�hl f�r die Jahreszeit�, hub er
    h�ndereibend an. �Sie kommen vom Lande herein, wie ich sehe, Sir?�

    �Woran sehen Sie denn das?� fragte Noah Claypole.

    �Wir haben nicht in London soviel Staub, wie Sie mitbringen�, erwiderte
    der Jude, nach Noahs und Charlottes Schuhen und den B�ndeln hinzeigend.

    �Sie sind mir ein pfiffiger Gesell�, versetzte Noah mit Lachen. �H�r'
    nur an, was er sagt, Charlotte.�

    �Ei nun, mein Lieber, man mu� wohl sein pfiffig in dieser Stadt�,
    fuhr der Jude, vertraulich fl�sternd, mit dem Finger an die Nase
    schlagend, fort, -- eine Geste, die Noah sogleich nachahmte, doch
    nicht mit vollst�ndigem Gelingen, da seine Nase nicht gro� genug dazu
    war. Fagin schien jedoch den Versuch so auszulegen, als wenn ihm Noah
    vollkommen h�tte beipflichten wollen, und schob dem letzteren sehr
    freundschaftlich den soeben von Barney kredenzten Krug zu.

    �Gutes Getr�nk�, entgegnete Fagin. �Wer es will immer trinken, mu�
    immer leeren etwas, eine Ladenkasse, eine Tasche, einen Strickbeutel,
    ein Haus, eine Postkutsche oder eine Bank.�

    Mr. Claypole sank r�ckw�rts auf seinen Stuhl und wandte sein kreidewei�
    gewordenes und grenzenlos best�rztes Gesicht vom Juden nach Charlotten.

    �Seien Sie ohne Sorgen meinetwegen, mein Lieber�, sagte Fagin, n�her
    r�ckend. �Ha, ha, ha! -- es war ein Gl�ck, da� niemand Sie h�rte, als
    ich zuf�llig -- es war ein gro�es Gl�ck f�r Sie.�

    �Ich nahm's nicht heraus�, stotterte Noah, die F��e nicht mehr wie ein
    unabh�ngiger Gentleman ausstreckend, sondern so tief unter den Stuhl
    ziehend, als er konnte. �Sie hat's ganz allein getan, und du hast's
    Charlotte; du wei�t, da� du's hast.�

    �Es ist gleichviel, mein Lieber, wer es hat oder wer es tat�, fiel der
    Jude ein, doch nichtsdestoweniger mit Falkenaugen nach dem M�dchen und
    den beiden B�ndeln hinblickend. �'s ist mein Gesch�ft auch, und Sie
    gefallen deswegen mir.�

    �Was ist Ihr Gesch�ft?� fragte Noah, sich einigerma�en wieder fassend.

    �Nun, dasselbe, das angefangen haben Sie,� antwortete Fagin, �und die
    Wirtsleute hier treiben es auch. Sie sind eingegangen zur rechten T�r
    und sind hier so sicher wie in Abrahams Scho�. Es gibt kein sichereres
    Haus in der Stadt als die Kr�ppel; das hei�t, wenn ich's will, und ich
    habe gefa�t eine Neigung zu Ihnen und dem jungen Frauenzimmer. Sie
    wissen nun Bescheid und k�nnen sich beruhigen vollkommen.�

    Noah blickte ihn trotz dieser Versicherung noch immer furchtsam und
    argw�hnisch an und r�ckte unruhig auf seinem Stuhle hin und her. Fagin
    nickte Charlotte freundlich zu, sprach ihr leise Mut zu und fuhr fort:
    �Ich will Ihnen sagen noch mehr. Ich hab' einen Freund, der Ihren
    Herzenswunsch, glaub' ich, kann befriedigen und Ihnen Gelegenheit
    geben, zu arbeiten vorerst in dem Gesch�ftszweige, der Ihnen gef�llt am
    besten, und Sie lehren alle anderen.�

    �Sie sprechen, als wenn es Ihr Ernst w�re,� bemerkte Noah.

    �Wenn ich nicht spr�che im Ernst, welchen Nutzen k�nnt' ich haben
    davon?� versetzte der Jude achselzuckend. �Kommen Sie -- lassen Sie
    mich reden mit Ihnen ein W�rtchen drau�en!�

    �Es tut nicht not, da� wir uns die M�he geben, hinauszugehen�, sagte
    Noah, die Beine allm�hlich wieder unter dem Stuhle hervorziehend. �Sie
    kann unterdes das Reisegep�ck in unsere Kammer tragen. Charlotte,
    bring' die B�ndel hinauf!�

    Charlotte gehorchte dem mit gro�er W�rde gegebenen Befehle ohne die
    mindeste Z�gerung, hob die beiden B�ndel auf und ging hinaus.

    �Hab' ich sie nicht ganz gut in der Zucht?� fragte Noah im Tone eines
    W�rters, der ein wildes Tier gez�hmt hat.

    �Oh, vortrefflich�, erwiderte Fagin, ihn auf die Schulter schlagend.
    �Sie sind ein Genie, mein Lieber.�

    �W�rde schwerlich hier sein, wenn ich's nicht w�re�, versetzte Noah.
    �Doch verlieren Sie keine Zeit, denn sie wird bald wieder da sein.�

    �Sehr wohl! Was meinen Sie -- wenn Ihnen gefiele mein Freund, was
    k�nnten Sie tun Besseres, als zu treten mit ihm in Verbindung?� sagte
    Fagin.

    �Es kommt darauf an, ob er gute Gesch�fte macht�, entgegnete Noah, dem
    Juden mit dem einen seiner kleinen Augen pfiffig zublinzelnd.

    �Er besch�ftigt eine Menge Leute und hat die beste Gesellschaft von
    allen, die treiben das Gesch�ft.�

    �Echte Stadtbursche?�

    �'s ist kein Nicht-Lond'ner drunter, und er w�rde Sie nicht einmal
    annehmen, selbst auf meine Empfehlung nicht, wenn es ihm nicht fehlte
    eben jetzt an Gehilfen.�

    �W�rd' ich 'rausr�cken m�ssen?� fragte Noah, an seine
    Beinkleidertaschen schlagend.

    �Ohne zwanzig Pfund ging's an unm�glich�, erwiderte Fagin auf das
    bestimmteste.

    �Aber zwanzig Pfund -- 's ist ein Haufen Geld!�

    �Eine Kleinigkeit, wenn Sie nicht k�nnen los werden die Banknote.�

    �Wann k�nnt' ich Ihren Freund sehen?�

    �Morgen fr�h.�

    �Wo?�

    �Hier.�

    �Hm! -- Wie hoch ist der Lohn?�

    �Sie leben wie ein Gentleman -- haben Kost und Wohnung und Tabak und
    Branntwein frei -- die H�lfte von allem, was Sie verdienen und was das
    junge Frauenzimmer verdient.�

    Es ist sehr zweifelhaft, ob Noah Claypole, so sehr bedeutend seine
    Habgier auch war, auf diese gl�nzenden Bedingungen eingegangen sein
    w�rde, wenn er h�tte vollkommen frei handeln k�nnen; allein er
    bedachte, da� es, wenn er nein sagte, in der Gewalt seines neuen
    Bekannten stand, ihn augenblicklich den H�nden der Gerechtigkeit zu
    �berliefern. Er erkl�rte daher, da� ihm der Vorschlag Fagins nicht ganz
    unannehmbar erschiene.

    �Aber sehen Sie,� setzte er hinzu, �da sie imstande ist, ein gutes
    St�ck Arbeit auszurichten, so m�cht' ich etwas recht Leichtes zugeteilt
    bekommen. Was w�rde jetzt wohl f�r mich passen? Es m��te so etwas sein,
    wobei ich mich nicht eben anzustrengen brauche und wobei keine Gefahr
    w�re.�

    �Mein Freund braucht jemand, der was Rechtes leisten k�nnte im
    Spionierfache -- was sagen Sie dazu?� entgegnete der Jude.

    �Gef�llt mir nicht ganz �bel, und bisweilen m�cht' ich wohl darin
    arbeiten�, sagte Noah z�gernd; �aber es wirft nur f�r sich selber
    nichts ab, wissen Sie.�

    �Freilich�, pflichtete Fagin ihm bei. �Was sagen Sie zu den alten
    Damen? Ihnen nehmen die Strickbeutel und Pakete und dann laufen um die
    Ecke -- damit wird gemacht viel Geld.�

    �Schreien diese aber nicht f�rchterlich, oder kratzen auch bisweilen?�
    wandte Noah kopfsch�ttelnd ein. �Ich habe keine Lust dazu. Ist kein
    anderes Fach offen?�

    �Halt, ja!� sagte der Jude, die Hand auf Noahs Knie legend. �Das
    Schratzchenbehandeln!�

    �Was ist denn das?�

    �Die Schratzchen sind die kleinen Kinder, die mit Sixpencen und
    Schillingen ausgeschickt werden von ihren M�ttern, um einzuholen
    allerhand; und das Behandeln ist wegnehmen ihnen das Geld, das sie
    immer haben in den H�nden, und sie dann sto�en in die Stra�enrinne und
    ganz langsam davongehen, als wenn geschehen w�re nichts, als da� ein
    Kind w�re gefallen und h�tte sich Schaden getan ein wenig. Ha, ha, ha!�

    �Ha, ha, ha!� stimmte Mr. Claypole ein und warf au�er sich vor
    Vergn�gen die F��e hoch in die Luft. �Beim Deuker, ja, das ist das
    Rechte!�

    �Gewi�, gewi߻, sagte Fagin; �Sie k�nnen haben prachtvolle Bezirke
    in Camden-Town und Battle-Bridge und solchen Gegenden mehr, wo immer
    ausgeschickt werden viele und zu jeder Tagesstunde niederwerfen so
    manche Schr�tzchen, wie Sie wollen nur.�

    �Ich bin alles wohl zufrieden�, sagte Noah, als er sich von seiner
    Ekstase wieder erholt hatte und Charlotte zur�ckgekehrt war. �Welche
    Zeit bestimmen wir auf morgen?�

    �Nun, belieben Sie zehn Uhr?�

    Noah nickte.

    �Welchen Namen soll ich nennen meinem Freunde?�

    �Mr. Bolter; Mr. Morris Bolter -- dies ist Mrs. Bolter.�

    �Ich bin Mrs. Bolters gehorsamer Diener�, sagte Fagin, sich mit
    grotesker Galanterie verbeugend. �Ich hoffe, Sie recht bald noch besser
    kennen zu lernen.�

    �H�rst du, was der Herr sagt, Charlotte?� herrschte ihr Mr. Claypole zu.

    �Ja, lieber Noah�, antwortete Charlotte, die Hand ausstreckend.

    Mr. Morris Bolter, sonst Claypole, wandte sich zu dem Juden und sagte:
    �Noah ist der Schmeichelname, den sie mir gibt.�

    �Oh, ich verstehe -- verstehe vollkommen�, erwiderte Fagin, f�r diesmal
    die Wahrheit redend. �Gute Nacht! Gute Nacht!�




    43. Kapitel.

        In welchem berichtet wird, wie sich der gepfefferte Baldowerer in
        Verlegenheiten benahm.


    �Also Ihr selber waret Euer Freund -- nicht wahr?� fragte Mr. Bolter,
    sonst Claypole, als er, nach zwischen ihm und Fagin besiegeltem
    Vertrage, in des Juden Wohnung gef�hrt worden war. �Dummkopf, der ich
    bin -- ich h�tt's mir doch gestern abend schon denken k�nnen!�

    �Jedermann ist sein eigener Freund�, erwiderte Fagin. �Es gibt
    Tausendk�nstler, die da sagen, Nummer Drei w�re die Zauberzahl, und
    andere sagen Nummer Sieben. Aber es ist nicht wahr, mein Freund. Nummer
    Eins ist's!�

    �Ha, ha, ha! Nummer Eins f�r immer!�

    �In einer kleinen Genossenschaft, wie die unsrige ist,� sagte der Jude,
    der eine Erkl�rung f�r n�tig hielt, �haben wir eine allgemeine Nummer
    Eins; das will sagen, Ihr k�nnt Euch nicht betrachten als Nummer Eins,
    ohne mich und all die anderen jungen Leute als dieselbe zu betrachten
    zugleich.�

    �Das w�r' der Teufel!�

    �Ihr seht wohl,� fuhr der Jude fort, sich anstellend, als ob er die
    Unterbrechung nicht beachtete, �unser Nutzen und Schaden ist eins so
    ganz, da� es nicht sein kann anders. Zum Beispiel, es ist Euer Zweck
    und Absicht, zu sorgen f�r Nummer Eins -- das hei�t f�r Euch selbst.�

    �Ganz recht, ganz recht.�

    �Sehr wohl -- Ihr k�nnt aber nicht sorgen f�r Euch selber, Nummer Eins,
    ohne zugleich zu sorgen f�r mich, Nummer Eins.�

    �Nummer Zwei wollt Ihr sagen�, fiel Mr. Bolter ein, der die Tugend der
    Selbstliebe im allerh�chsten Ma�e besa�.

    �Nein, nein!� entgegnete der Jude. �Ich bin von derselben Wichtigkeit
    f�r Euch, wie Ihr es seid selbst.�

    �H�rt,� unterbrach Mr. Bolter, �Ihr seid ein sehr netter Mann, und ich
    halte sehr viel von Euch; aber so dicke Freunde, wie Ihr mit dem allen
    meint, sind wir doch noch nicht.�

    �Bedenkt doch, bedenkt doch nur!� sagte der Jude achselzuckend und die
    H�nde ausstreckend. �Ihr habt getan, was sehr h�bsch war, und ich ehre
    und liebe Euch deshalb; aber 's ist auch derart, da� es Euch zugleich
    einbringen kann die Krawatte, die so leicht ist einzukn�pfen und so
    schwer wieder aufzubinden -- den Strick n�mlich!�

    Mr. Bolter legte die Hand an sein Halstuch, als wenn es ihm unbequem
    eng s��e, und murmelte eine Art von Beistimmung.

    �Der Galgen,� fuhr Fagin fort, �der Galgen, mein Lieber, ist ein
    h��licher Wegweiser, der zeigt um eine sehr scharfe Ecke und hat
    gemacht ein Ende der Weiterreise vieler mutvoller, wackerer Leute auf
    der gro�en Heerstra�e. Euch zu halten auf der bequemen Stra�e und zu
    bleiben dem Galgen fern, mu� sein Euer Nummer Eins, mein Lieber.�

    �Nat�rlich�, fiel Mr. Bolter ein; �aber wozu redet Ihr von solchen
    Dingen?�

    �Blo� um Euch zu zeigen meine Meinung deutlich�, erwiderte Fagin, die
    Augenbrauen emporziehend. �Ihr k�nnt das nicht allein, sondern h�ngt
    dabei ab von mir, und ich h�nge ab von Euch, wenn mein kleines Gesch�ft
    soll haben guten Fortgang. Das erste ist Eure Nummer Eins, das zweite
    ist meine Nummer Eins. Je mehr Euch liegt am Herzen Eure Nummer Eins,
    desto mehr m��t Ihr sein besorgt f�r meine; und so kommen wir endlich
    wieder zur�ck auf das, was ich Euch sagte gleich anfangs -- da� Sorge
    f�r Nummer Eins kommt uns allen zugut, und lassen wir's fehlen daran,
    gehen wir zugrunde miteinander alle.�

    �Das ist wohl wahr�, bemerkte Bolter gedankenvoll. �Ihr seid, meiner
    Treu, ein geriebener alter Gesell!�

    Fagin erkannte mit innigstem Vergn�gen, da� dies keine blo�e
    Schmeichelei war, sondern da� er seinem Rekruten eine bedeutende
    Vorstellung von seiner Verschlagenheit und Gewalt beigebracht hatte,
    was beim Beginn ihrer beiderseitigen Bekanntschaft von gro�er
    Wichtigkeit war. Um den Eindruck, den er auf den jungen Menschen
    gemacht hatte, noch zu verst�rken, lie� er ihn einige Blicke in
    die Gro�artigkeit und den Umfang seiner Operationen tun, wobei er,
    seinem Zwecke gem��, Wahrheit und Dichtung so geschickt miteinander
    vermischte, da� Mr. Bolters Hochachtung gegen ihn sichtlich zunahm und
    er zugleich eine Zutat heilsamer Furcht erhielt, welche bei ihm zu
    erwecken �u�erst w�nschenswert war.

    �Dies gegenseitige Vertrauen ist es,� sagte der Jude, �was mich tr�stet
    wegen schwerer Verluste. Erst gestern morgen verlor ich meinen besten
    Gehilfen.�

    �Ist er Euch davongegangen?� fragte Mr. Bolter.

    �Ganz wider seinen Willen�, antwortete Fagin. �Er war beschuldigt des
    Versuchs eines Taschendiebstahls, und sie fanden bei ihm eine silberne
    Schnupftabaksdose. Es war seine eigene, mein Lieber, seine eigene, denn
    er schnupfte selbst, und die Dose war ihm sehr wert. Er ward wieder
    vorbeschieden auf heute, denn sie meinten herbeischaffen zu k�nnen den
    Eigent�mer. Oh, er war wert f�nfzig silberne Dosen, und ich w�rde sie
    darum geben, wenn ich ihn h�tte wieder. Ihr solltet gekannt haben den
    Baldowerer, mein Lieber; solltet gekannt haben den Gepfefferten!�

    �Ich hoffe ihn noch kennen zu lernen -- meint Ihr nicht, Fagin?�

    �Ich mu� es bezweifeln�, erwiderte der Jude seufzend. �Wenn vorgebracht
    wird kein neues Zeugnis gegen ihn, so werden wir ihn freilich haben
    wieder nach ein sechs oder acht Wochen; sonst aber wird er gerumpelt,
    und auf lebenslang, sicher auf lebenslang; denn sie wissen's, welch ein
    gescheiter Bursch ist der Baldowerer.�

    �Was wollt Ihr damit sagen, da� er gerumpelt w�rde?� fragte Mr. Bolter.
    �Warum sprecht Ihr in solchen Ausdr�cken zu mir, da Ihr doch wi�t, da�
    ich sie nicht verstehen kann?�

    Fagin war im Begriff, ihm zu sagen, da� Rumpeln soviel als Deportieren
    bedeute, allein in demselben Augenblicke trat Master Bates mit den
    H�nden in den Beinkleidertaschen und einem halbkomisch-tr�bseligen
    Gesichte herein.

    �'s ist vorbei mit ihm, Fagin�, sagte er, nachdem er seinem neuen
    Kameraden geb�hrend vorgestellt worden war.

    �Was willst du sagen damit?� fragte der Jude mit bebenden Lippen.

    �Sie haben den Herrn ausgesp�rt, dem die Dose geh�rte, und noch mehrere
    Anklagen vorgebracht -- der Gepfefferte erh�lt freie �berfahrt�,
    antwortete Master Bates. �Ich mu� � vollst�nd'gen Traueranzug haben,
    Fagin, und � Hutband, ihn zu besuchen, eh' er seine Reise antritt.
    's is doch die M�glichkeit! -- Jack Dawkins -- der gro�e Jack Dawkins
    -- der Baldowerer -- der gepfefferte Baldowerer -- und wird gerumpelt
    wegen 'ner lumpigen Schnupftabaksdose! -- wegen 'ner erb�rmlichen
    Dorfdruckerei[AS]. Ich h�tt's nimmermehr geglaubt, da� er's unter 'ner
    goldenen Uhr mit Kette und Petschaften zum mind'sten getan haben w�rde.
    Nein, wenn er noch 'nem reichen alten Herrn seine ganze Massumme[AT]
    und alles geganft[AU] h�tte, so da� er doch abreiste wie � Gentleman!
    -- aber so -- wie � gemeiner Dorfdrucker! -- ohne Ruhm, ohne Ehre!�

      [AS] Taschendiebstahl.

      [AT] Geld.

      [AU] geraubt.

    Also seine Gef�hle f�r den ungl�cklichen Freund ausdr�ckend, nahm
    Master Bates entr�stet und niedergeschlagen auf dem ersten besten
    Stuhle Platz.

    �Was schw�tzest du, da� er h�tte weder Ruhm noch Ehre!� rief Fagin,
    seinem Z�glinge einen zornigen Blick zuwerfend, aus. �Ist er nicht
    immer gegangen �ber euch allen -- hat's einer von euch ihm tun k�nnen
    gleich -- nur von fern tun k�nnen gleich -- wie?�

    �Freilich, freilich! Aber sollt's einen denn nicht jammern,� entgegnete
    Charley, �sollte man nicht des Teufels werden vor Verdru�, da� nichts
    davon vor Gericht verlautet, da� niemand nur zur H�lfte erf�hrt,
    wer und was er gewesen ist? Welch 'nen elenden Titel wird er im
    Newgatekalender bekommen -- kommt vielleicht nicht mal 'nein! O weh, o
    weh, was es f�r � Jammer ist!�

    �Ah! wenn du's so meinst,� sagte der Jude mit vergn�gtem Kichern und
    ihm die Hand reichend, �wenn du's so meinst, das ist ein andres.
    Schaut, mein Lieber,� fuhr er, zu Bolter sich wendend, fort, �schaut,
    wie stolz sie sind auf ihren Stand und Beruf! Ist es nicht zu sehen
    eine Lust?�

    Mr. Bolter nickte Beistimmung, und der Jude trat mit freudigem Stolze
    zu Charley, klopfte ihm auf die Schulter und sagte tr�stend: �Sei nur
    ohne Sorgen, Charley; es wird schon kommen an den Tag, und er wird's
    selbst schon zeigen, was er ist gewesen, und wird keine Schande bringen
    �ber seine alten Kameraden und Lehrer. Bedenkt auch nur, wie jung er
    noch ist! Ist's nicht schon eine gro�e Auszeichnung, bei seinen Jahren
    gerumpelt zu werden auf lebenslang?�

    �Ja freilich -- daran hatt' ich nicht gedacht -- 's ist freilich
    ehrenvoll genug!� erwiderte Charley, einigerma�en getr�stet.

    �Er wird haben alles, was er braucht�, fuhr der Jude fort; �wird in
    Doves[AV] gehalten werden wie ein Gentleman, alle Tage haben sein Bier
    und Geld in der Tasche, zu spielen Bild oder Schrift, wenn er's nicht
    kann ausgeben.�

      [AV] Gef�ngnis.

    �Wahr und wahrhaftig?� rief Charley aus.

    �Ganz gewi�, ganz gewi�!� sagte Fagin. �Und wir werden ihm schaffen
    'nen Advokaten -- den zungenfertigsten, der wird sein zu finden -- zu
    f�hren seine Verteid'gung; und wenn er will, kann er auch halten eine
    Rede selbst, und wir werden's lesen alles in den Bl�ttern. Was sagst
    du, Charley?�

    �Pr�chtig, pr�chtig!� rief Master Bates aus. �Oh, es ist mir, als
    wenn ich ihn vor mir s�he, wie er die alten Per�cken bei der Nase
    herumzieht, wie sie sich abstrap'zieren, wichtig und feierlich
    aussehen, und er so vertraulich und gem�tlich zu ihnen spricht, als
    wenn er des Richters eigener Sohn w�re und 'ne Rede bei Tisch hielte --
    ha, ha, ha!�

    �Aber Charley,� sagte der Jude, �wir m�ssen ersinnen, ein Mittel in
    Erfahrung zu bringen, wie er sich macht heute und was ihm passiert.�

    �Soll ich hingehen?� entgegnete Charley eifrig, denn er versprach sich
    jetzt den k�stlichsten Genu� von einem Schauspiele, bei welchem der
    Baldowerer, den er noch vor kurzem als einen Gegenstand des Mitleides
    und Verdrusses betrachtet, in der ersten gl�nzenden Rolle auftreten
    sollte.

    �Nicht um alles in der Welt�, antwortete Fagin.

    �So schickt den da -- den Neuangeworbenen hin�, riet Charley; �den
    kennt niemand.�

    �Kein schlechter Rat�, sagte der Jude. �Was meint Ihr, mein Lieber?�

    �Nein, nein�, erwiderte Mr. Bolter kopfsch�ttelnd; �nichts davon, 's
    ist mein Fach nicht.�

    �Was habt Ihr ihm denn f�r � Fach zugeteilt, Fagin?� fragte Charley
    Bates, Noahs schlottrige Gestalt mit gro�em Widerwillen betrachtend.
    �Sich den R�cken zu decken, wenn was zu riskieren ist, und alles
    aufzuessen, wenn wir in guter Ruhe zu Haus sitzen?�

    �Geht dich nichts an�, fiel Mr. Bolter ein; �und nimm dir keine
    Freiheiten heraus gegen Leute, die �ber dir sind, Knirps, oder du wirst
    erfahren, da� du vor die unrechte Schmiede gekommen bist.�

    Master Bates belachte die prahlerische Drohung so ausgelassen, da� es
    einige Zeit w�hrte, bevor Fagin vermitteln und Mr. Bolter vorstellen
    konnte, da� er bei einem Besuche des Polizeiamts durchaus keine Gefahr
    liefe; denn von seiner kleinen Aff�re w�rde noch ebensowenig Kunde nach
    der Hauptstadt, wo man ihn am wenigsten vermute, gelangt sein, wie
    ein Steckbrief; und sollte es das Ungl�ck gewollt haben, so w�rde er
    sich, gut verkleidet, nirgends in ganz London mit gr��erer Sicherheit
    aufhalten k�nnen, als eben auf der Polizei, wo er ohne Zweifel am
    letzten gesucht werden d�rfte.

    Mr. Bolter lie� sich endlich durch diese und �hnliche Vorstellungen,
    noch mehr aber durch seine Furcht vor dem Juden bewegen, freilich mit
    der verdrie�lichsten Miene, einzuwilligen, die Sendung zu �bernehmen.
    Fagin versah ihn sogleich mit einem K�rrnerkittel, manchesternen
    Kniehosen, ledernen Beinlingen, einem Hut mit Weggeldzetteln und
    einer Peitsche, und zweifelte um so weniger am Erfolge, da Mr. Bolter
    obendrein die Ungelenkheit eines K�rrners im vollkommensten Ma�e besa�.
    Der Baldowerer wurde ihm genau beschrieben, und Master Bates geleitete
    ihn durch Nebengassen nach Bow-Street, wies ihn zurecht, erteilte ihm
    jede sonst n�tige Auskunft, forderte ihn zur Eile auf und versprach,
    seine R�ckkehr an der Stelle, wo er ihn verlie�, zu erwarten.

    Master Bates' Weisungen waren so genau gewesen, da� sich Noah Claypole
    oder Morris Bolter, wie der Leser will, sehr leicht, und ohne fragen
    zu m�ssen, zurechtfand. Er dr�ngte sich durch einen haupts�chlich
    aus Frauenzimmern bestehenden Haufen hinein in das d�stere und
    schmutzige Gerichtszimmer. Vor den Schranken standen ein paar Weiber,
    die ihren bewundernden Angeh�rigen oder Bekannten zunickten, w�hrend
    der Gerichtsschreiber zwei Polizisten und einem einfach gekleideten,
    �ber den Tisch lehnenden Manne Zeugenaussagen vorlas und ein
    Gef�ngnisw�rter l�ssig dastand und von Zeit zu Zeit Ruhe oder �das Kind
    hinauszuschaffen� gebot, wenn ein ungeb�hrliches Gefl�ster oder der
    Aufschrei eines S�uglings eine St�rung verursachte. Noah blickte scharf
    umher nach dem Baldowerer, bemerkte Leute genug, welche Geschwister
    oder Eltern des Gepfefferten h�tten sein k�nnen, aber niemand, auf den
    die Beschreibung gepa�t h�tte, die ihm von Jack Dawkins selbst gegeben
    worden war. Endlich waren die vor den Schranken stehenden Frauenzimmer
    abgeurteilt und entfernt, und nunmehr erschien ein Angeklagter, der
    ohne Frage der Baldowerer war.

    Jack ging vor dem Gef�ngnisw�rter, den Hut in der rechten Hand haltend
    und die Linke in der Beinkleidertasche, keck genug einher und fragte,
    sobald er auf der Anklagebank stand, sogleich mit h�rbarer Stimme,
    warum man ihn an die schimpfliche Stelle gef�hrt habe.

    �Willst du wohl den Mund halten?� rief ihm der Schlie�er zu.

    �Bin ich kein Engl�nder?� rief der Baldowerer zur�ck. �Wo sind meine
    Freiheiten?�

    �Wirst sie bald genug bekommen�, entgegnete der Schlie�er, �und zwar
    mit Pfeffer dazu.�

    �Je nun, wenn sie mir gekr�nkt werden, so wird sich's schon
    finden, was der Staatssekret�r f�r die inneren Angelegenheiten den
    Oberschenkeln[AW] zu sagen hat�, fuhr Jack Dawkins fort. �Jetzo aber --
    holla, was gibt's hier? Wollen die Friedensrichter nicht so gut sein,
    diese kleine Sache abzumachen und mich nicht aufzuhalten, indem sie die
    Zeitungen lesen? Ich hab' 'nen Gentleman nach der City bestellt, bin
    ein Mann von Wort und auch sehr p�nktlich in Gesch�ften; er wird daher
    fortgehen, wenn ich nicht zur bestimmten Zeit da bin, und es k�nnte
    'ne Klage auf Schadenersatz geben gegen die, die mich aufgehalten
    haben. -- He, Binnfaden[AX], wie hei�en die beiden Abrosche[AY] da
    auf der Richterbank?� wandte er sich zu dem Gef�ngnisw�rter, was die
    N�chststehenden derma�en kitzelte, da� sie fast so herzlich lachten,
    wie es Master Bates selbst getan haben w�rde, wenn er die spa�hafte
    Frage geh�rt h�tte.

      [AW] Richtern.

      [AX] Amtsdiener.

      [AY] Spitzbuben.

    �Ruhe da!� rief der Schlie�er.

    Einer der Friedensrichter fragte nach der Ursache des Ger�usches.

    �Hier steht ein Taschendieb, Ihr Edlen.�

    �Ist der Knabe schon hier gewesen?�

    �H�tt's schon manchmal sein sollen, Ihr Edlen. �berall sonst ist er
    schon lange genug gewesen. Ich kenne ihn sehr wohl, Ihr Edlen.�

    �So! Ihr kennt mich also?� rief der Baldowerer, sich anstellend, als
    wenn er die Angabe aufzeichnete. �Sehr wohl. Das setzt 'ne Klage wegen
    Beschimpfung meines guten Namens.�

    Es wurde abermals gelacht und abermals Ruhe geboten.

    �Wo sind die Zeugen?� begann der Gerichtsschreiber.

    �Ach, so ist's recht!� fiel Jack Dawkins ein. �Ja, wo sind die Zeugen?
    Ich m�chte doch das Pl�sier haben, sie zu sehen!�

    Sein Wunsch wurde augenblicklich erf�llt, denn es trat ein Polizist
    vor, der gesehen hatte, da� der Angeklagte einem Herrn das Taschentuch
    aus der Tasche gezogen, und da es ein sehr altes gewesen, nachdem er
    Gebrauch davon gemacht, wieder hineingesteckt hatte. Er hatte deshalb
    den T�ter verhaftet und bei demselben eine silberne Schnupftabaksdose
    mit dem Namen des Eigent�mers auf dem Deckel gefunden. Der Eigent�mer
    der Dose war gleichfalls gegenw�rtig, beschwor, da� die Dose die
    seinige w�re, und da� er sie vermi�t h�tte, sobald er sich Bahn aus
    dem Gedr�nge gemacht, in welchem (wie sich fand) der Angeklagte das
    fragliche Taschentuch entwendet und zur�ckgegeben. Er hatte auch
    bemerkt, da� sich ein junger Gentleman eiligst von ihm entfernt, und
    der junge Gentleman war eben der Baldowerer.

    �Hast du eine Frage an den Zeugen zu richten, Knabe?� fragte der
    Friedensrichter.

    �Ich mag mich nicht erniedrigen, mit ihm in Unterredung zu treten�,
    entgegnete Jack Dawkins.

    �Hast du �berhaupt was zu sagen?�

    �H�rst du die Frage Seiner Edlen nicht, ob du etwas zu sagen h�ttest?�
    fiel der Schlie�er, den stummen Baldowerer mit dem Ellbogen ansto�end,
    ein.

    �Bitt' um Vergebung�, sagte Jack, zerstreut aufblickend. �Redeten Sie
    mich an?�

    �Ihr Edlen,� bemerkte der Schlie�er, �ich hab' mein Lebtag noch keinen
    solchen jungen Erzspitzbuben gesehen. Willst du was sagen, Bursch?�

    �Nein,� entgegnete der Baldowerer, �hier nicht; dies ist nicht das
    rechte Kaufhaus f�r die Gerechtigkeit, und au�erdem fr�hst�ckt
    mein Advokat heute morgen bei dem Vizepr�sidenten des Hauses der
    Gemeinen. Jedoch werden wir, ich und er und eine sehr reputierliche
    Bekanntschaft, anderw�rts sprechen, und zwar so, da� die
    Richterper�cken w�nschen werden, da� sie niemals geboren oder da� sie
    von ihren Bedienten aufgeh�ngt sein m�chten, statt mich hier heute
    morgen zu prozessieren. Ich will --�

    �Er ist vollst�ndig �berf�hrt; ins Gef�ngnis mit ihm -- man bringe ihn
    hinaus!� rief der Gerichtsschreiber.

    �Komm her, Bursch�, sagte der Schlie�er.

    �Komme schon�, sagte der Baldowerer, seinen Hut mit der flachen Hand
    gl�ttend, und wandte sich darauf nach der Richterbank: �Es hilft Ihnen
    nichts, Gentlemen, und wenn Sie auch noch so best�rzt aussehen. Ich
    werde kein Erbarmen mit Ihnen haben, f�r keinen Heller nicht. Sie
    werden daf�r b��en, und ich m�chte um vieles nicht an Ihrer Stelle
    sein. Ich w�rde die Freiheit nicht annehmen, und wenn Sie mich auf den
    blo�en Knien darum anflehten. Binnfaden, f�hr mich ab ins Gef�ngnis!�

    Der Schlie�er zog ihn beim Kragen heraus, Jack drohte, die Sache
    vors Parlament zu bringen, und l�chelte darauf den Schlie�er mit der
    behaglichsten Selbstzufriedenheit an.

    Sobald ihn Noah hatte fortschleppen sehen, eilte er zu Master Bates
    zur�ck, der ihn in einem angemessenen Verstecke erwartet hatte und sich
    zeigte, sobald er sich vergewissert, da� niemand seinem neuen Bekannten
    nachfolgte. Sie gingen schleunigst miteinander nach Hause, um Fagin
    die erfreuliche Kunde zu bringen, da� sich der Baldowerer vollkommen
    ehrenhaft benommen und sich einen gl�nzenden Namen gemacht habe.




    44. Kapitel.

        Nancy wird verhindert, ihr Rose Maylie gegebenes Versprechen zu
        erf�llen.


    Wie vollkommen eingeweiht Nancy in alle Verstellungsk�nste auch war,
    vermochte sie doch die Gem�tsbewegungen nicht g�nzlich zu verbergen,
    die das Bewu�tsein ihres Schrittes bei ihr hervorbrachte. Sie erinnerte
    sich, da� sowohl der listige Jude wie der brutale Sikes sie in das
    Geheimnis von Anschl�gen, die sie vor allen anderen verborgen hielten,
    eingeweiht hatten, und zwar im vollkommensten Vertrauen auf ihre Treue
    und �ber allen Verdacht erhabene Zuverl�ssigkeit; und so sch�ndlich
    jene Anschl�ge, so ruchlos die Urheber derselben sein mochten, so
    erbittert sie selbst gegen den Juden war, der sie Schritt f�r Schritt
    tiefer und immer tiefer in einen Abgrund von Verbrechen und Elend
    gef�hrt hatte, aus welchem kein Entrinnen m�glich war: es gab doch
    Augenblicke, wo bei ihr eine mildere Stimmung gegen ihn vorherrschte
    und der Gedanke ihr Unruhe verursachte, da� ihn endlich infolge der
    von ihr gemachten Enth�llung sein lange vermiedenes, aber freilich
    vollkommen verdientes Schicksal ereilen m�chte.

    Doch waren dies nur vor�bergehende Gedanken und Gef�hle bei ihr, deren
    sie sich aus Macht der Gewohnheit nicht g�nzlich zu erwehren imstande
    war; denn ihr Entschlu� stand fest, und ihr Charakter war derart,
    da� sie sich durch keinerlei R�cksichten bewegen lie�, einen einmal
    gefa�ten Entschlu� wieder aufzugeben. Ihre Besorgnis f�r Sikes w�rde
    ein noch st�rkerer Beweggrund gewesen sein, zur�ckzutreten, solange es
    noch Zeit war; allein sie hatte es sich ausbedungen, da� ihr Geheimnis
    streng bewahrt werden sollte -- hatte keinen Faden an die Hand gegeben,
    der zu seiner Entdeckung f�hren konnte -- hatte um seinetwillen sogar
    das Anerbieten einer Zuflucht vor allem sie umgebenden Laster und
    Elend zur�ckgewiesen -- und was konnte sie mehr tun? Sie war und blieb
    entschlossen.

    Obgleich aber alle ihre inneren K�mpfe so endeten, erneuerten sie sich
    doch fortw�hrend und lie�en auch ihre Spuren zur�ck. Nach wenigen
    Tagen sah sie bla� und abgezehrt aus. Bisweilen beachtete sie gar
    nicht, was um sie her vorging, und nahm an Gespr�chen keinen Teil, bei
    welchen sie sonst die Lebhafteste und Lauteste gewesen sein w�rde; und
    bisweilen lachte sie wieder ohne Heiterkeit und l�rmte ohne Zweck und
    Veranlassung. Zu anderen Zeiten -- und oft einen Augenblick darauf --
    sa� sie schweigend, niedergeschlagen, hinbr�tend, den Kopf auf die
    H�nde gest�tzt, da, w�hrend gerade die Anstrengung, womit sie sich dann
    wieder aufraffte, noch st�rker verk�ndete, da� sie Unruhe empfand und
    da� ihre Gedanken mit ganz anderen Dingen als denen besch�ftigt waren,
    die von ihren Gesellschaftern besprochen wurden.

    Der Sonntagabend war gekommen, und die Glocke der n�chsten Kirche
    schlug elf. Sikes und der Jude unterbrachen ihr Gespr�ch und horchten
    -- und aufblickend und noch gespannter horchte Nancy.

    �'ne Stunde vor Mitternacht�, sagte Sikes, das Fenster �ffnend und nach
    seinem Stuhle zur�ckkehrend; �auch ist's neblig und finster -- 'ne gute
    Gesch�ftsnacht.�

    �Ah, ja,� sagte Fagin, �'s ist sehr schade, Bill, da� es eben nichts
    gibt zu tun.�

    �Da hast du mal recht�, entgegnete Sikes barsch. �'s ist um so mehr
    schade, da ich obendrein recht in der Laune dazu bin.�

    Der Jude sch�ttelte seufzend den Kopf.

    �Wir m�ssen die verlorene Zeit wieder einzubringen suchen, wenn wieder
    was Gutes eingef�delt ist�, fuhr Sikes fort.

    �So ist's recht, mein Lieber�, erwiderte Fagin, sich erdreistend, ihn
    auf die Schulter zu klopfen. �Es freut mich herzinnig, Euch reden zu
    h�ren so.�

    �Freut Euch herzinnig -- so! Meinetwegen�, sagte Sikes.

    �Ha, ha, ha!� lachte der Jude, als wenn ihm schon dies sehr geringe
    Zugest�ndnis Freude gew�hrte. �Ihr seid heute abend der echte,
    wahrhaftige Bill -- wieder ganz Ihr selber, mein Lieber.�

    �Mir ist's, als w�r ich ein ganz anderer, wenn du mir die alte, welke
    Tatze auf die Schulter legst -- runter damit!� rief Sikes, die Hand des
    Juden zur�ckschleudernd.

    �Wird Euch schlimm dabei, Bill -- erinnert's Euch ans Gefa�twerden?�
    fragte der Jude, entschlossen, keine Empfindlichkeit zu zeigen.

    �Ja -- aber ans Gefa�twerden vom Teufel, nicht von 'nem H�scher. Von
    Adam her ist kein Mensch gewesen mit 'nem Gesicht wie das deinige,
    m��te denn sein dein Vater, und dem wird wohl jetzund sein grauer Bart
    versengt, sofern nicht Satan selber dein Vater ist, was mich eben nicht
    wundern w�rde.�

    Fagin erwiderte nichts auf diese Schmeichelei, sondern zupfte Sikes
    am �rmel und wies nach Nancy hin, die ganz in der Stille den Hut
    aufgesetzt hatte und eben hinausgehen wollte.

    �Heda, Nancy!� rief Sikes. �Wohin will die Dirne bei dieser
    Nachtstunde?�

    �Nicht weit.�

    �Was ist das f�r 'ne Antwort! Wohin willst du?�

    �Ich sage, nicht weit.�

    �Und ich sage, wohin? Hast du geh�rt?�

    �Ich wei� nicht, wohin.�

    �Dann wei� ich's�, sagte Sikes, mehr aus Eigensinn, als da� er einen
    bestimmten Grund gehabt h�tte, sich Nancys Ausgehen, wohin es ihr
    beliebte, zu widersetzen. �Nirgend. Setz dich wieder hin.�

    �Ich bin unwohl, wie ich Euch schon gesagt habe, und mu� frische Luft
    sch�pfen.�

    �Steck den Kopf aus 'm Fenster 'naus, das ist ebensogut.�

    �Das ist's nicht; ich mu� Bewegung haben.�

    �So -- du sollst aber keine haben�, entgegnete Sikes, stand auf,
    verschlo� die T�r, zog den Schl�ssel aus, ri� dem M�dchen den Hut vom
    Kopfe und warf ihn auf einen alten Schrank. �Willst du jetzt ruhig
    dableiben, wo du bist, oder nicht?�

    �Ich kann auch ohne Hut gehen�, sagte Nancy erblassend. �Was soll dies
    bedeuten, Bill? Wi�t Ihr auch, was Ihr tut?�

    �Ob ich wei�, was -- Fagin, sie ist von Sinnen, denn sie w�rde sich's
    sonst nicht herausnehmen, solche Worte zu mir zu sprechen!�

    �Ihr macht's danach, da� ich etwas Verzweifeltes tue�, murmelte Nancy,
    beide H�nde gegen die Brust pressend, als wenn sie einen heftigen
    Ausbruch gewaltsam zur�ckdr�ngen wollte. �La�t mich hinaus -- in dieser
    Minute -- diesem Augenblick --�

    �Nein!� schrie Sikes.

    �Fagin, sagt ihm, da� er mich gehen l��t. Ich rat's ihm. H�rt Ihr?�
    rief Nancy, mit den F��en stampfend.

    �Ob ich dich h�re? Ja�, rief Sikes zur�ck; �und wenn ich dich noch
    ein paar Augenblicke h�re, so soll dich der Hund derma�en an der Kehle
    packen, da� er dir die kreischende Stimme herausrei�t. Was f�llt dir
    ein, Weibsbild -- was steckt dir im Kopfe?�

    �La�t mich gehen�, sagte Nancy flehend, setzte sich an die T�r auf den
    Boden nieder und fuhr fort: �Bill, la�t mich gehen; Ihr wi�t nicht, was
    Ihr tut -- wi�t's wahrlich nicht. Nur eine -- nur eine einzige Stunde.�

    �Ich will mich vierteln lassen,� rief Sikes, sie sehr unsanft beim Arm
    fassend, �wenn ich nicht glaube, da� die Dirne verr�ckt -- toll und
    verr�ckt geworden ist. Steh auf!�

    �Ich stehe nicht eher auf, als bis Ihr mich gehen la�t -- nicht eher!�
    schrie Nancy.

    Sikes blickte sie eine Weile an, ersah den rechten Augenblick, fa�te
    pl�tzlich ihre beiden H�nde, zog die Str�ubende in ein ansto�endes
    Gemach, setzte sich auf eine Bank, warf sie auf einen Stuhl und hielt
    sie gewaltsam nieder. Sie bat und suchte sich ihm abwechselnd mit
    Gewalt zu entziehen, gab endlich, als es zw�lf geschlagen hatte, ganz
    ersch�pft ihre Versuche auf, und Sikes verlie� sie mit einer durch
    mehrfache kr�ftige Beteuerungen unterst�tzten Warnung, um zu Fagin
    zur�ckzukehren.

    �Was f�r'n sonderbares Gesch�pf die Dirne ist!� sagte er, sich den
    Schwei� abwischend.

    �Das m�gt Ihr wohl sagen -- m�gt Ihr wohl sagen, Bill�, versetzte der
    Jude nachdenklich.

    �Was meinst du denn, was ihr im Kopfe gesteckt hat, noch so sp�t mit
    Gewalt ausgehen zu wollen? Du mu�t sie besser kennen als ich -- was
    meinst du, Jude?�

    �Eigensinn, glaub' ich -- Weibertrotz und Eigensinn, mein Lieber�,
    antwortete Fagin achselzuckend.

    �Glaub's auch. Ich dachte, da� ich sie zahm gemacht h�tte, sie ist aber
    so schlimm wie je.�

    �Noch schlimmer, Bill. Ich habe so etwas erlebt noch niemals an ihr,
    und um solch 'ner geringen Ursache.�

    �Ich auch nicht. Es scheint, mein Fieber steckt ihr im Blut und will
    nicht 'raus -- was?�

    �Mag wohl sein, Bill.�

    �Ich will ihr 'n bissel Blut abzapfen, ohn' den Doktor zu bem�hn, wenn
    sie's wieder so macht.�

    Der Jude nickte beistimmend.

    �Sie war Tag und Nacht um mich,� fuhr Sikes fort, �als ich auf der
    Seite lag, w�hrend du wie 'n falscher Kujon, der du bist, dich fern
    h�ltst. Wir hatten die ganze Zeit nichts zu bei�en und zu brechen,
    und ich glaub', es hat sie verdrie�lich gemacht, und sie ist unruhig
    geworden, weil sie so lang hat im Haus sitzen m�ssen -- he?�

    �Ganz recht, mein Lieber�, erwiderte Fagin fl�sternd. �Pst!�

    In diesem Augenblick trat Nancy wieder herein und setzte sich an ihren
    gewohnten Platz. Ihre Augen waren rot und geschwollen: sie wiegte sich
    hin und her, warf den Kopf empor und brach nach einiger Zeit in ein
    Gel�chter aus.

    �Was ist denn dies nun wieder?� rief Sikes, erstaunt zu Fagin sich
    wendend, aus.

    Der Jude gab ihm einen Wink, sie f�r den Augenblick nicht weiter zu
    beachten, und nach einigen Minuten sa� sie wieder da wie vorhin. Er
    fl�sterte Sikes zu, sie w�rde von nun an ganz ruhig bleiben, nahm
    seinen Hut und sagte ihm gute Nacht. An der T�r stand er still, drehte
    sich noch einmal um und bat, da� ihm jemand auf der dunkeln Treppe
    leuchten m�chte.

    �Leucht ihm 'nunter�, sagte Sikes, der eben seine Pfeife f�llte.
    �'s w�re schade, wenn er hier selbst den Hals br�che und den
    H�ngezuschauern nichts zu gaffen g�be.�

    Nancy geleitete den alten Mann mit dem Lichte hinunter. Auf dem
    Hausflur angelangt, legte er den Finger auf den Mund und fl�sterte ihr
    in das Ohr: �Was hattest du, liebes Kind?�

    �Wieso?� erwiderte sie, gleichfalls fl�sternd.

    �Warum du ausgehen wolltest mit Gewalt. Wenn er,� sagte Fagin, mit dem
    kn�chernen Finger nach oben zeigend, �wenn er ist so barbarisch gegen
    dich -- er ist ein Tier, Nancy, ein unvern�nftiges, wildes Tier --
    warum --�

    �Nun?� fragte sie, als er, den Mund dicht an ihrem Ohre und die Augen
    dicht vor den ihrigen, innehielt.

    �La� jetzt gut sein,� fuhr der Jude fort, �wollen ein andermal sprechen
    davon. Du hast einen Freund an mir, Kind, einen treuen Freund. Ich
    hab' auch die Mittel -- wenn du willst dich r�chen an ihm, der
    dich behandelt wie einen Hund -- schlimmer als einen Hund, dem er
    schmeichelt bisweilen doch -- so komm zu mir; komm zu mir, was ich dir
    sage. Er ist ein Tagesfreund; aber mich kennst du von alters her, Nancy
    -- von alters her.�

    �Ich kenne Euch sehr wohl�, antwortete das M�dchen, ohne die mindeste
    Bewegung zu zeigen. �Gute Nacht.�

    Sie trat zur�ck, als er ihr die Hand reichen wollte, sagte ihm aber
    noch einmal mit fester Stimme gute Nacht, erwiderte den Blick, den er
    ihr zum Abschiede zuwarf, mit einem hinl�ngliches Verstehen andeutenden
    Zunicken und verschlo� die T�r hinter ihm.

    Fagin kehrte gedankenvoll nach seiner Wohnung zur�ck. Er war schon
    seit einiger Zeit der Ansicht, in welcher ihn das soeben Vorgefallene
    best�rkte, da� Nancy der schlechten Behandlung, welche sie von dem
    brutalen Sikes erfuhr, m�de geworden sei und eine Neigung zu einem
    neuen Freunde gefa�t habe. Ihr ver�ndertes Wesen, da� sie so h�ufig
    allein ausging, ihre verh�ltnism��ige Gleichg�ltigkeit gegen den
    Vorteil oder Schaden der Bande, f�r welche sie vormals so gro�en Eifer
    bewiesen hatte, und dazu ihr so heftiges Verlangen, an diesem Abende
    und gerade zu einer bestimmten Stunde das Haus noch verlassen zu
    wollen: dieses alles unterst�tzte seine Annahme und �berzeugte ihn fest
    von der Richtigkeit derselben. Der Gegenstand dieser neuen Liebschaft
    des M�dchens befand sich unter den Leuten seines Anhangs nicht. Er
    mu�te mit einer Alliierten wie Nancy eine sch�tzbare Erwerbung sein,
    die es so bald wie m�glich zu machen galt.

    Auch war noch ein anderer und finsterer Zweck zu erreichen. Sikes
    wu�te zu viel, und seine plumpen, beleidigenden Reden hatten Fagin
    darum nicht minder verletzt und gereizt, weil er es sich nicht merken
    lie�. Nancy konnte es nicht entgehen, da� sie, sobald sie sich von
    ihm trennte, vor seiner Wut nicht sicher war und da� er dieselbe ohne
    allen Zweifel auch an ihrem neuen Liebhaber auslassen w�rde, so da�
    die gesunden Gliedma�en, ja das Leben desselben in offenbarer Gefahr
    schwebten. Fagin glaubte, sie w�rde sich leicht bereden lassen, ihn zu
    vergiften. �Weiber,� dachte er, �haben so etwas und noch Schlimmeres
    wohl schon getan, um die Ziele zu erreichen, die das M�dchen jetzt
    verfolgt. Tut sie es, so werde ich von dem gef�hrlichen Halunken,
    dem Menschen, den ich hasse, befreit -- erhalte einen Ersatzmann f�r
    ihn, und mein Einflu� �ber Nancy ist, bei meiner Kenntnis dieses
    Verbrechens, fortan ganz unbegrenzt.�

    Dies waren seine Gedanken gewesen, w�hrend ihn Sikes allein gelassen,
    und er hatte deshalb beim Fortgehen das M�dchen auszuforschen gesucht.
    Sie hatte keine �berraschung gezeigt, sich nicht angestellt, als ob sie
    ihn nicht verst�nde, vielleicht bewies der Blick, mit welchem sie ihm
    zum zweitenmal gute Nacht gesagt, klar, da� sie seine Meinung sehr wohl
    verstanden hatte.

    Aber sie weigerte sich vielleicht, in einen Anschlag auf Sikes Leben
    einzugehen, worauf es haupts�chlich ankam. �Wie kann ich meinen Einflu�
    bei ihr vergr��ern?� dachte der Jude auf seinem Heimwege. �Welche neue
    Gewalt �ber sie kann ich mir verschaffen?�

    Ein Gehirn, wie das seinige, ist fruchtbar an Hilfsmitteln. Sollte er
    sie nicht seinen Pl�nen f�gsam machen k�nnen, wenn er sie, sofern kein
    Gest�ndnis von ihr zu erlangen war, von einem Kundschafter beobachten
    lie�, den Gegenstand ihrer neuen Leidenschaft entdeckte und Sikes (vor
    dem sie sich in hohem Ma�e f�rchtete) alles zu enth�llen drohte, falls
    sie nicht einwilligte, zu tun, was er von ihr verlangte.

    �Es wird angehen�, sagte er fast laut; �hab' ich nur erst ihr
    Geheimnis, so darf sie mir's nicht abschlagen -- so gewi� ihr an ihrem
    Leben liegt. Ich besitze die Mittel, Nancy, und nur Geduld, Sikes, ich
    hab' euch, hab' euch beide!�

    Er wandte sich mit einer drohenden Handbewegung um und warf einen
    finsteren Blick nach der Stra�e zur�ck, wo er den verwegenen B�sewicht
    verlassen, und senkte im Weitergehen die kn�chernen H�nde in die Falten
    seines zerlumpten Mantels, die er zusammenpre�te, als wenn er einen
    verha�ten Feind zwischen den spitzigen Fingern h�tte.




    45. Kapitel.

        Noah Claypole wird von Fagin als Spion verwandt.


    Der alte Mann stand am anderen Morgen beizeiten auf und erwartete
    ungeduldig seinen neuen Verb�ndeten, der sich erst nach einem endlos
    scheinenden Ausbleiben zeigte und sogleich mit Gier �ber das Fr�hst�ck
    herfiel.

    �Bolter�, sagte der Jude, sich ihm gegen�ber setzend.

    �Was gibt's?� erwiderte Noah. �Fordert nichts von mir, bis ich mit'm
    Essen fertig bin. Das ist der gro�e Fehler hier. Es wird einem niemals
    Zeit genug bei den Mahlzeiten gelassen.�

    �Ei, Ihr k�nnt doch sprechen beim Essen�, sagte der Jude, vom Grunde
    seines Herzens des jungen Freundes E�gier verw�nschend.

    �Und es geht obendrein noch besser, wenn ich spreche�, versetzte
    Bolter, ein ungeheures St�ck Brot abschneidend. �Wo steckt denn
    Charlotte?�

    �Ich habe sie ausgeschickt heute morgen mit dem anderen jungen
    Frauenzimmer, weil ich w�nschte zu sein allein.�

    �Wollte nur, da� Ihr der Dirne erst gesagt h�ttet, sie sollte
    Brotschnitte mit Butter r�sten. Nun schwatzt aber nur zu -- werde mich
    nicht st�ren lassen�, sagte Noah, und es schien in der Tat wenig auf
    sich zu haben mit der Besorgnis, da� er sich st�ren lassen d�rfte, denn
    er war offenbar entschlossen, wacker fortzuarbeiten.

    �Ihr habt gestern gemacht Eure Sachen gut�, sagte der Jude; �sehr
    sch�n. Sechs Schillinge, neun Pence und 'nen halben Penny am
    allerersten Tage! Das Schratzchen wird Euch machen reich.�

    �Verge�t nicht die drei Bierkannen und den Milchtopf�, erwiderte Bolter.

    �Nein, nein, mein Lieber. Die Bierkannen waren gro�e Geniebeweise, der
    Milchtopf aber war ein vollkommenes Meisterst�ck.�

    �Ging wohl an f�r 'nen Anf�nger�, bemerkte Mr. Bolter selbstgef�llig.
    �Die Bierkannen nahm ich von 'nem Sout'raingitter 'runter, und der
    Milchtopf stand drau�en vor 'nem Gasthofe; ich dachte also, er m�chte
    rostig werden durch den Regen oder sich erk�lten, wi�t Ihr. Ha, ha, ha!�

    Der Jude stimmte in Mr. Bolters Gel�chter, der seine Besch�ftigung
    r�stig weiter fortsetzte, des Scheines halber herzlich ein und sagte,
    sich �ber den Tisch hin�berlehnend: �Ihr m��t mir ausrichten etwas,
    mein Lieber, das erfordert gro�e Sorgfalt und Vorsicht.�

    �Fagin,� entgegnete Noah, �Ihr d�rft aber nichts Gef�hrliches von mir
    verlangen und mich nicht wieder in Polizeigerichte schicken; denn ein
    f�r allemal, das gef�llt mir nicht, und ich will's nicht.�

    �'s ist dabei nicht die geringste Gefahr -- Ihr sollt blo� baldowern
    ein Frauenzimmer.�

    �Ein altes?�

    �Ein junges.�

    �Nun, darauf versteh' ich mich gut genug -- trieb's schon mit Gl�ck,
    als ich noch in die Schule ging. Was soll ich denn auskundschaften von
    der jungen Person?�

    �Wohin sie geht, mit wem sie verkehrt, und wom�glich, was sie sagt;
    Euch merken die Stra�e, wenn's eine Stra�e, oder das Haus, wenn's ist
    ein Haus, und mir bringen so viel Kunde, wie Ihr nur verm�gt.�

    �Was gebt Ihr mir daf�r?� fragte Noah begierig.

    �Wenn Ihr's gut ausrichtet, ein Pfund, mein Lieber -- ja, ja, ein
    Pfund�, erwiderte Fagin, der ihn so sehr wie m�glich f�r die Sache zu
    interessieren w�nschte; �und das ist so viel, als ich noch nie habe
    gegeben f�r ein St�ck Arbeit, wobei nicht war viel zu gewinnen.�

    �Wer ist denn das Frauenzimmer?�

    �Eine der Unsern.�

    �Hm -- so! Ihr setzt Mi�trauen in sie?�

    �Sie hat sich gewandt zu neuen Liebhabern, und ich mu� wissen, wer die
    m�gen sein.�

    �Verstehe schon -- um das Vergn�gen zu haben, sie kennen zu lernen,
    wenn es respektable Leute sind -- wie? Ha, ha, ha! Verla�t Euch auf
    mich.�

    �Wu�te wohl, da� ich's w�rde k�nnen.�

    �Nat�rlich, nat�rlich! Wo ist sie? Wo mu� ich ihr auflauern? Wann geh'
    ich los?�

    �Ihr sollt das alles h�ren von mir, mein Lieber, zu seiner Zeit. Haltet
    Euch bereit nur und �berla�t das �brige mir.�

    Sechs Abende sa� der Kundschafter gestiefelt und in seinem
    K�rrneranzuge da, bereit, auf einen Wink von Fagin zu beginnen, und
    Abend f�r Abend kehrte der Jude verdrie�lich nach Hause zur�ck und
    sagte, da� es noch nicht Zeit w�re. Am siebenten -- einem Sonntagabend
    -- trat er mit einem Vergn�gen ein, das er nicht zu verbergen imstande
    war.

    �Sie geht aus heute abend,� sagte er, �und ich bin gewi�, da� sie geht
    hin da, wo ist zu erforschen, was ich w�nsche zu wissen; denn sie
    hat allein gesessen den ganzen Tag, und der Mann, vor dem sie sich
    f�rchtet, wird erst zur�ckkehren gegen Tagesanbruch. Kommt, kommt!
    Folgt mir geschwind!�

    Des Juden Erregtheit steckte auch Noah an, der sogleich aufsprang. Sie
    verlie�en das Haus, eilten durch ein Stra�en- und Gassenlabyrinth und
    langten endlich vor einem Gasthause an, in welchem Noah die Kr�ppel
    erkannte. Es war elf Uhr vor�ber und die T�r verschlossen; sie �ffnete
    sich aber auf ein leises Pfeifen des Juden und schlo� sich wieder, als
    sie ger�uschlos hineingegangen waren. Fagin fl�sterte kaum, sondern
    besprach sich mit dem j�dischen J�nglinge durch stumme Zeichen, wies
    darauf nach dem kleinen Fenster hin und bedeutete Noah, auf einen
    Stuhl zu steigen und sich die im ansto�enden Zimmer befindliche Person
    anzusehen.

    �Ist das das Frauenzimmer?� fl�sterte Noah. �Sie sieht vor sich nieder,
    und das Licht steht hinter ihr. Ich kann ihr Gesicht nicht erkennen.�

    �Bleibt ruhig stehen�, fl�sterte Fagin und gab Barney ein Zeichen, der
    sogleich hinausging, nach ein paar Augenblicken in dem ansto�enden
    Zimmer erschien, das Licht, unter dem Vorwande, es zu schneuzen, vor
    das Frauenzimmer -- Nancy -- hinstellte, sie anredete und dadurch
    veranla�te, den Kopf emporzuheben.

    �Jetzt seh' ich sie�, fl�sterte Noah.

    �Deutlich?�

    �W�rde sie unter Tausenden wiedererkennen.�

    Nancy stand auf und schickte sich zum Fortgehen an. Er stieg eilig von
    dem Stuhle herunter und trat sacht mit Fagin hinter einen Vorhang;
    gleich darauf ging Nancy durch das Zimmer und aus dem Hause hinaus.

    �Pst!� rief Barney, der ihr die Haust�r ge�ffnet, �jetzt.�

    Noah wechselte einen Blick mit Fagin und schl�pfte hinaus.

    �Links�, fl�sterte Barney; �haltet Euch linker Hand und auf der anderen
    Seite.�

    Noah sah Nancy beim Laternenscheine schon in einiger Entfernung. Er
    eilte ihr nach, folgte ihr so nahe, wie es ihm r�tlich erschien, und
    hielt sich auf der anderen Seite, um sie desto besser beobachten zu
    k�nnen. Sie sah sich �ngstlich ein paarmal um und stand einmal still,
    um einige ihr dicht nachfolgende M�nner vor�berzulassen. Sie schien im
    Weitergehen Mut zu gewinnen und einen sicheren und festeren Schritt
    anzunehmen. Der Kundschafter hielt sich in gemessener Entfernung hinter
    ihr und lie� sie nicht aus den Augen.




    46. Kapitel.

        Nancy erf�llt ihre Zusage.


    Die Kirchenglocken schlugen dreiviertel auf elf Uhr, als zwei Gestalten
    auf der Londoner Br�cke erschienen. Die eine leicht und rasch vorw�rts
    eilende war die eines M�dchens, das unruhig um sich blickte, als
    erwarte sie jemand; die andere die eines Mannes, der im tiefsten
    Schatten, den er finden konnte, der ersteren in einiger Entfernung
    nachschlich, aber stillstand, wenn das M�dchen stillstand, und wieder
    vordrang, so schnell oder langsam dasselbe sich eben fortbewegte. So
    schritten sie �ber die Br�cke von dem Middlesex- nach dem Surreyufer
    hin�ber. Das M�dchen, das alle Vor�bergehenden mit forschenden Blicken
    gemustert hatte, schien sich in seiner Erwartung get�uscht zu haben,
    drehte sich pl�tzlich um und ging wieder zur�ck. Der Kundschafter war
    indes auf seiner Hut gewesen, trat in eine Vertiefung, lehnte �ber das
    Gel�nder, lie� das M�dchen vor�ber und folgte ihr sodann wieder nach
    wie vorher. Fast mitten auf der Br�cke angelangt, stand sie still und
    er gleichfalls.

    Es war eine sehr finstere und kalte Nacht, nur wenige gingen an den
    beiden vor�ber und beachteten sie nicht. Die Themse war von dichtem
    Nebel bedeckt, den der matte, r�tliche Glanz der Feuer auf den
    kleinen, in den Werften ankernden Fahrzeugen kaum zu durchdringen
    vermochte, und die Feuer lie�en die H�user am Ufer nur als d�mmrige,
    noch undeutlichere Massen erscheinen. Die T�rme der alten Heilands-
    und St.-Magnus-Kirche -- so lange schon die riesigen W�chter der
    alten Br�cke -- waren sichtbar durch die Finsternis, der Wald der
    Schiffsmaste aber unter der Br�cke und weiter umher die Menge der T�rme
    auch f�r den sch�rfsten Blick unerkennbar.

    Das M�dchen war -- fortw�hrend von seinem ungesehenen Beobachter
    verfolgt -- unruhig ein paarmal hin und wieder �ber die Br�cke
    gegangen, als die Glocke der St.-Pauls-Kirche abermals das
    Hinscheiden eines Tages verk�ndete. Mitternacht war gekommen �ber
    die menschenerf�llte Stadt, die Pal�ste und H�tten, die Bettler- und
    Gaunerh�hlen, den Kerker und das Irrenhaus, die Gem�cher, in welchen
    neues Leben begann und abgelaufenes endete, Gesunde ruhten und Kranke
    �chzten, Leichen starr dalagen und bl�hende Kinder s�� schlummerten und
    tr�umten.

    Nicht zwei Minuten, nachdem der letzte Glockenton verklungen war,
    stiegen eine junge Dame und ein grauk�pfiger Herr aus einem Mietswagen
    nicht weit von der Br�cke, auf welche sie rasch zuschritten. Sie hatten
    sich kaum auf derselben gezeigt, als das M�dchen aufmerksam stillstand
    und ihnen sodann entgegeneilte, deren Munde ein Ausruf der �berraschung
    entfloh, welchen sie jedoch sogleich unterdr�ckten, als ein wie ein
    K�rrner Gekleideter pl�tzlich fast gegen sie anrannte.

    �Nicht hier�, fl�sterte Nancy hastig. �Ich f�rchte mich, hier mit Ihnen
    zu reden. Folgen Sie mir dort die Treppe hinunter.�

    Der K�rrner drehte sich um, w�hrend sie so sprach und nach der Treppe
    hinwies, rief in rauhem Tone zur�ck, �wozu sie die Breite des ganzen
    Steinpflasters einn�hmen�, und ging vor�ber.

    Die Treppe, nach welcher das M�dchen hingewiesen hatte, befand sich am
    Surreyufer und f�hrte zu einem Landungsplatze hinunter; der K�rrner
    eilte hin zu ihr, blickte forschend umher und fing an, hinabzusteigen.
    Sie besteht aus drei Abs�tzen, auf deren zweitem die Mauer linker Hand
    in einen Pfeiler nach der Themse hin ausl�uft. Die Stufen der unteren
    Flucht sind breiter, und wer nur um eine einzige tiefer hinter den
    Pfeiler tritt, ist denen verborgen, die, wenn auch ganz in seiner N�he,
    auf dem Treppenabsatze stehen. An dieser Stelle versteckte sich der
    K�rrner, mit dem R�cken an den Pfeiler tretend. Er war in gespanntester
    Erwartung, denn was hier vorging, lag g�nzlich au�er dem Kreise aller
    seiner Vermutungen, und wollte schon wieder h�her hinaufgehen, als er
    den Schall von Fu�tritten und gleich darauf dicht neben sich Stimmen
    vernahm. Er horchte mit verhaltenem Atem.

    �Dies ist weit genug�, sagte der Herr. �Ich lasse die junge Dame nicht
    weiter hinuntergehen. Viele andere w�rden Ihnen nicht einmal so weit
    gefolgt sein; Sie sehen, da� ich Ihnen Vertrauen bewiesen habe.�

    �Sie sind in der Tat sehr vorsichtig -- oder auch mi�trauisch, wie mir
    scheint. Doch gleichviel�, sagte Nancy.

    �Weshalb f�hren Sie uns denn aber an einen solchen Ort?� fragte der
    Herr in einem milderen Tone. �Warum wollten Sie sich nicht dort oben
    sprechen lassen, wo es hell ist und wo doch Menschen in der N�he sind?�

    �Ich habe es Ihnen schon gesagt, da� ich mich f�rchtete, dort mit
    Ihnen zu reden. Ich wei� nicht, wie es kommt,� sagte Nancy schaudernd,
    �bin aber so beklommen und zittere so sehr, da� ich kaum auf den F��en
    stehen kann.�

    �Was f�rchten Sie denn?� fragte der Herr mitleidig.

    �Ich wei� es selbst kaum�, erwiderte das M�dchen. �Den ganzen Tag
    haben mich schreckliche Gedanken an die verschiedensten Todesarten und
    blutige Leichent�cher heimgesucht, und fortw�hrend hat mich eine Angst
    gequ�lt, da� es mir war, als wenn ich mitten im Feuer brannte. Ich las
    heute abend in einem Buche, um mir die Zeit zu verk�rzen, und las nur
    immer dasselbe heraus.�

    �Einbildungen�, sagte der alte Herr beruhigend.

    �Nein, nein�, entgegnete das M�dchen mit heiserer Stimme. �Ich will
    darauf schw�ren, da� das Wort >Sarg< auf jeder Seite des Buches mit
    gro�en, schwarzen Lettern gedruckt stand -- und erst vor kurzem, als
    ich hierher ging, ward einer dicht an mir vor�bergetragen.�

    �Das ist nichts Ungew�hnliches. Ich habe sehr oft S�rge an mir
    vor�bertragen sehen.�

    �*Wirkliche* -- das war aber dieser nicht.�

    Sie sprach dies alles in einem Tone, da� es den versteckten Lauscher
    kalt �berlief, ja, da� ihm das Blut in den Adern erstarrte. Er
    hatte nie eine gr��ere Herzenserleichterung empfunden, als in dem
    Augenblicke, da er die s��e Stimme der jungen Dame -- Roses -- vernahm,
    die Nancy bat, sich zu beruhigen und sich nicht so entsetzlichen
    Gedanken hinzugeben.

    �Reden Sie ihr freundlich zu, der Armen; sie scheint es zu bed�rfen�,
    f�gte sie, zu ihrem Begleiter sich wendend, hinzu.

    �Ihre hochm�tigen, frommen Damen w�rden mich, wenn sie mich in dieser
    Nacht s�hen, wie ich bin, ver�chtlich anblicken und mir vom ewigen
    H�llenfeuer und der Rache des Himmels predigen�, rief das M�dchen
    aus. �Oh, meine teure junge Dame, warum sind die nicht, die Gottes
    Auserw�hlte sein wollen, so mild und g�tig gegen uns arme Ungl�ckliche
    wie Sie! Ach, Sie besitzen alles, was jene verloren haben, Jugend und
    Sch�nheit, und k�nnten gar wohl ein wenig stolz sein, statt so viel
    bescheidener.�

    �Ah!� fiel der Herr ein; �ein T�rke kehrt sein Antlitz, nachdem er es
    reinlich abgewaschen, nach Osten, indem er seine Gebete spricht. Jene
    guten Leute reiben an der rauhen Welt die Freundlichkeit von ihren
    Gesichtern ab und wenden sie dann nach der finsteren Seite des Himmels.
    Hab' ich zwischen dem Muselman und Pharis�er zu w�hlen, so lobe ich mir
    den ersteren.�

    Er sprach die Worte zu der jungen Dame, doch vielleicht beabsichtigend,
    Nancy Zeit zu verschaffen, sich wieder zu sammeln. Bald darauf redete
    er das M�dchen an.

    �Sie waren am vorigen Sonntagabend nicht hier.�

    �Ich konnte nicht kommen -- wurde gewaltsam zur�ckgehalten.�

    �Von wem?�

    �Von Bill -- dem Manne, von dem ich der jungen Dame erz�hlt habe.�

    �Ich will doch hoffen, da� niemand Verdacht wegen der Sache auf Sie
    geworfen hat, die uns jetzt zusammengef�hrt?� fragte der alte Herr
    besorgt.

    �Nein�, antwortete Nancy kopfsch�ttelnd. �Es ist aber nicht eben leicht
    f�r mich, ihn zu verlassen, ohne da� er wei�, warum, und ich w�rde auch
    zu der Dame nicht haben gehen k�nnen, h�tt' ich ihm nicht, um mich von
    ihm entfernen zu k�nnen, einen Schlaftrunk gegeben.�

    �War er denn vor Ihrer R�ckkehr erwacht?�

    �Nein; und so wenig, wie er selbst, hat sonst jemand Verdacht auf mich
    geworfen.�

    �Gut. H�ren Sie mich jetzt an. Diese junge Dame hat mir und einigen
    andern, das vollkommenste Zutrauen verdienenden Freunden mitgeteilt,
    was Sie vor vierzehn Tagen ihr anvertraut haben. Ich gestehe, anfangs
    Zweifel gehegt zu haben, ob man sich ganz auf Ihre Aussagen verlassen
    k�nnte, halte mich aber jetzt davon �berzeugt.�

    �Das k�nnen Sie allerdings sein�, beteuerte Nancy.

    �Ich wiederhole, da� ich es fest glaube; und um Ihnen zu beweisen, da�
    ich Ihnen zu vertrauen geneigt bin, sage ich Ihnen ohne R�ckhalt, da�
    wir das Geheimnis, worin es auch bestehen mag, Monks durch Furcht zu
    entrei�en gesonnen sind. Doch wenn -- wenn wir seiner nicht sollten
    habhaft werden k�nnen, oder wenn ihm nichts abzudringen w�re, so m�ssen
    Sie uns den Juden in die H�nde liefern.�

    �Fagin!� rief das M�dchen, pl�tzlich zur�cktretend, aus.

    �Ihn -- ja, ihn m�ssen Sie uns in die H�nde liefern.�

    �Nimmermehr -- das werd' ich nimmermehr tun�, entgegnete Nancy; �werde
    es nie tun, solch ein Teufel er auch ist, und obwohl er �rger als ein
    Teufel an mir gehandelt hat.�

    �Sie wollen also nicht?� fragte der Herr, der keine andere Antwort
    erwartet zu haben schien.

    �In keinem Falle!�

    �Dann sagen Sie mir, warum Sie es nicht wollen.�

    �Aus einem Grunde,� erwiderte Nancy mit Festigkeit, �aus einem Grunde,
    den die Dame kennt, und ich wei�, denn ich habe ihr Versprechen, sie
    wird dabei auf meiner Seite stehen; und aus dem weiteren Grunde, weil
    ich -- ein so ruchloses Leben er auch gef�hrt hat -- gleichfalls einen
    schlechten Wandel gef�hrt habe. Viele von uns sind miteinander schlecht
    und b�se gewesen, und ich will sie nicht verraten, die mich h�tten
    verraten k�nnen und es, so schlecht sie sind, nicht taten.�

    �Dann,� fiel der Herr lebhaft ein, als wenn er erreicht h�tte, was er
    eben gewollt, �dann liefern Sie mir Monks in die H�nde und �berlassen
    Sie es mir, nach Gutd�nken mit ihm zu verfahren.�

    �Wie aber, wenn er die andern verr�t?�

    �Ich verspreche Ihnen, da� die Sache ruhen soll, sobald wir ihm die
    Wahrheit abgerungen haben. In Olivers kleiner Lebensgeschichte kommen
    ohne Zweifel Umst�nde vor, die man nur sehr ungern der �ffentlichkeit
    preisgeben w�rde, und ist nur die Wahrheit heraus, so soll niemand in
    Ungelegenheit kommen.�

    �Aber wenn Sie sie nicht herausbekommen?�

    �Dann soll der Jude nicht ohne Ihre Einwilligung vor Gericht gezogen
    werden, und ich glaube Ihnen f�r den Fall Gr�nde vorlegen zu k�nnen,
    nach deren Anh�ren Sie einwilligen werden.�

    �Hab' ich daf�r das Versprechen der Dame?� fragte Nancy mit Nachdruck.

    �Ich gebe es Ihnen�, nahm Rose das Wort; �gebe Ihnen die aufrichtigste
    und bestimmteste Zusage.�

    �Monks soll nie erfahren, wie Sie zu der Kunde, die Sie durch
    mich besitzen, gelangt sind?� fuhr das M�dchen nach einem kurzen
    Stillschweigen fort.

    �Nein -- nie,� erwiderte der Herr. �Er soll es nicht einmal vermuten
    k�nnen.�

    �Ich bin eine L�gnerin gewesen und habe unter L�gnern gelebt von meiner
    fr�hsten Kindheit an, will aber Ihren Worten Glauben schenken�, sagte
    Nancy nach einem abermaligen Stillschweigen.

    Beide versicherten ihr, da� sie es getrost k�nne, und nunmehr nannte
    sie ihnen, so leise fl�sternd, da� der Horcher nur sehr schwer zu
    verstehen vermochte, den Namen, den Stadtteil und die Stra�e der
    Taverne, aus welcher ihr Noah nach der Br�cke gefolgt war. Sie sprach
    in kurzen Pausen; der Herr schien sich das N�tigste zu notieren. Als
    sie auch das Innere des Hauses beschrieben und angegeben hatte, wie
    es am besten beobachtet werden k�nnte, und an welchen Abenden und zu
    welchen Stunden es von Monks besucht zu werden pflegte, schien sie ein
    paar Augenblicke innezuhalten, um sich die Z�ge und das ganze �u�ere
    desselben um so lebhafter zur�ckzurufen.

    �Er ist gro�,� sagte sie, �und kr�ftig gebaut, aber nicht stark; er hat
    einen lauernden Gang und blickt beim Gehen best�ndig erst �ber die eine
    und dann �ber die andere Schulter. Vergessen Sie das nicht, denn seine
    Augen liegen so tief wie nur immer m�glich im Kopfe, so da� Sie ihn
    daran fast allein schon unter Tausenden zu erkennen verm�gen. Er hat
    dunkles Haar und Augen und ein schw�rzliches Gesicht, das aber �ltlich
    und verfallen aussieht, obwohl er nicht �ber sechs- bis achtundzwanzig
    Jahre alt sein kann. Seine Lippen sind oft blau und durch Bisse
    entstellt, denn er hat f�rchterliche Zuf�lle und bei�t sich bisweilen
    sogar die H�nde blutig -- warum stutzen Sie?� fragte sie pl�tzlich
    abbrechend.

    Der Herr erwiderte hastig, da� er sich dessen nicht bewu�t w�re, und er
    bat sie, fortzufahren.

    �Ich mu�te dies gro�enteils von andern herauslocken, um es Ihnen sagen
    zu k�nnen,� sprach das M�dchen weiter, �denn ich habe ihn nur zweimal
    gesehen, und beide Male war er in einen weiten Mantel eingeh�llt. Mehr
    glaube ich Ihnen nicht -- doch ja! An seinem Halse, so hoch hinauf, da�
    man etwas davon sehen kann, wenn er sein Gesicht abwendet, ist --�

    �Ein breites rotes Mal, wie von einer Brandwunde�, fiel der Herr ein.

    �Wie -- Sie kennen ihn?� rief das M�dchen aus.

    Roses Lippen entfloh ein Ausruf des h�chsten Erstaunens, und auf einige
    Augenblicke waren alle drei so stumm, da� der Horcher sie atmen h�ren
    konnte.

    �Ich glaube es�, unterbrach der Herr jedoch bald das Stillschweigen.
    �Nach Ihrer Beschreibung sollte ich ihn allerdings kennen. Wir werden
    indes sehen. Es gibt auffallende �hnlichkeiten -- kann sein, da� er
    dennoch ein anderer ist.�

    Er trat bei diesen, in einem verstellt gleichg�ltigen Tone gesprochenen
    Worten ein paar Schritte zur�ck, wobei er sich dem versteckten
    Kundschafter n�herte, der ihn fl�stern h�rte: �Er mu� es sein!� Gleich
    darauf sagte er wieder laut: �Junges M�dchen, Sie haben uns die
    wichtigsten Dienste geleistet, und ich w�nsche Ihnen dankbar daf�r zu
    sein. Was kann ich f�r Sie tun?�

    �Nichts�, erwiderte Nancy.

    �So d�rfen Sie nicht sprechen�, fuhr der Herr in einem so dringenden
    und herzlichen Tone fort, da� auch ein weit verh�rteteres Gem�t dadurch
    h�tte ger�hrt werden m�gen. �Ich bitte, sagen Sie es mir.�

    �Ich mu� dabei bleiben, Sir�, entgegnete Nancy weinend. �Sie k�nnen
    nichts tun, mir zu helfen. F�r mich ist wahrlich keine Hoffnung �brig.�

    �Sie schneiden sich die Hoffnung selbst ab�, fuhr der Herr fort. �Ihre
    Vergangenheit ist eine beklagenswerte Verschwendung unsch�tzbarer
    Jugendgaben gewesen, wie sie der Sch�pfer nur einmal gibt und nicht
    wieder verleiht; auf die Zukunft aber k�nnen Sie Hoffnung setzen.
    Ich sage nicht, da� es in unserer Macht stehe, Ihnen Seelenfrieden
    zu bieten, der Ihnen nur in dem Ma�e werden kann, wie Sie selbst ihn
    suchen; wohl aber sind wir imstande, und es ist unser eifriger Wunsch,
    Ihnen einen stillen Zufluchtsort entweder im Lande oder, wenn Sie
    Furcht hegen, hierzubleiben, au�er Landes zu verschaffen. Noch ehe
    der Morgen graut, sollen Sie Ihren bisherigen Genossen so g�nzlich
    entr�ckt sein und so wenige Spuren hinter sich zur�cklassen, als wenn
    Sie von der Erde verschwunden w�ren. Geben Sie unseren Vorstellungen
    und Bitten nach. Ich m�chte nicht, da� Sie auch nur noch ein einziges
    Wort mit den Leuten Ihres bisherigen Umgangs wechselten, nur noch einen
    Blick auf die St�tte Ihres bisherigen Daseins w�rfen, oder die Luft
    nur wieder atmeten, welche Pest und Tod f�r Sie ist. Geben Sie unsern
    Bitten nach, w�hrend es noch Zeit ist, solange Sie noch k�nnen.�

    �Sie l��t sich bewegen�, rief Rose aus; �ich wei� es, sie fa�t den
    rettenden Entschlu�.�

    �Nein, nein�, erwiderte Nancy nach einem kurzen inneren Kampfe; �ich
    bin an mein bisheriges Leben gekettet. Ich verabscheue und hasse es
    jetzt, kann es aber nicht aufgeben. Ich war schon l�ngst zu weit
    gegangen, um zur�ckkehren zu k�nnen -- und doch wei� ich nicht, ob ich
    es nicht versucht haben w�rde, wenn Sie vor einiger Zeit so zu mir
    gesprochen h�tten. Doch diese Angst ergreift mich wieder,� setzte sie,
    sich hastig umwendend, hinzu: �ich mu� nach Hause gehen.�

    �Nach Hause?!� wiederholte Rose, gro�en Nachdruck auf die Worte legend.

    �Nach Hause, Mi� -- nach einem solchen Hause, wie ich es mir durch die
    ganze M�he meines Lebens erbaut habe. Lassen Sie uns scheiden. Man wird
    mich beobachten oder sehen. Fort, fort von hier! Habe ich Ihnen einen
    Dienst geleistet, so erzeigen Sie mir nur die einzige G�te, zu gehen
    und mich allein nach Hause zur�ckkehren zu lassen.�

    �Es ist vergeblich�, sagte der Herr seufzend. �Wir gef�hrden vielleicht
    Ihre Person, wenn wir hier weilen, und haben Sie vielleicht schon
    l�nger aufgehalten, als Sie erwartet haben.�

    �Ja, ja, das haben Sie�, sagte Nancy.

    �Was kann das Ende des Lebens der �rmsten sein?� rief Rose aus.

    �Schauen Sie hinunter in das finstere Wasser!� sagte das M�dchen.
    �Wie oft lesen Sie von meinesgleichen, die sich in die Fluten
    hinunterst�rzen und kein lebendes Wesen, sie zu beweinen oder nur nach
    ihnen zu fragen, zur�cklassen. Es k�nnen Jahre dar�ber hingehen oder
    vielleicht nur Monate, doch nicht besser wird zuletzt mein Ende sein.�

    �O bitte, reden Sie nicht so�, sagte Rose schluchzend.

    �Sie werden nie davon h�ren, beste junge Dame, und Gott verh�te, da�
    solcher Graus -- Gute Nacht, gute Nacht!�

    Der Herr wandte sich ab.

    �Nehmen Sie um meinetwillen diese B�rse,� sagte Rose, �damit es Ihnen
    in der Stunde der Not nicht an einer Hilfsquelle mangle.�

    �Nein, nein�, entgegnete das M�dchen. �Ich habe, was ich tat, nicht f�r
    Geld getan. Lassen Sie mir dieses Bewu�tsein. Doch -- geben Sie mir ein
    Andenken -- etwas, das Sie getragen haben -- nein, nein, keinen Ring --
    Ihre Handschuhe oder Ihr Taschentuch -- so, des Himmels Segen �ber Sie
    -- gute Nacht, gute Nacht!�

    Ihre heftige Erregtheit und die Besorgnis einer Entdeckung, welche
    gef�hrlich f�r sie werden k�nnte, bewog den Herrn, sich ihrem Verlangen
    gem�� mit Rose zu entfernen. Auf der obersten Stufe angelangt, standen
    beide still.

    �Horch!� fl�sterte Rose. �Rief sie nicht? Ich glaube, da� ich ihre
    Stimme h�rte!�

    �Nein, mein liebes Fr�ulein�, erwiderte Brownlow, traurig
    zur�ckblickend. �Sie hat sich nicht einmal leise geregt und wird es
    auch nicht eher, als bis wir fort sind.�

    Rose z�gerte noch immer, allein der alte Herr legte ihren Arm in den
    seinigen und zog sie mit sanfter Gewalt fort. Sobald sie verschwunden
    waren, warf sich Nancy fast der L�nge nach auf eine der Treppenstufen
    nieder und machte ihrer Herzensqual durch bittere Tr�nen Luft. Nach
    einiger Zeit stand sie wieder auf und begann mit wankenden Schritten
    ihren Heimweg. Der erstaunte Horcher blieb noch einige Minuten hinter
    dem Pfeiler stehen, schlich sodann die Treppe hinauf, lugte vorsichtig
    umher und eilte dann, so schnell er konnte, nach dem Hause des Juden
    zur�ck.




    47. Kapitel.

        Ungl�ckliche Folgen.


    Es war fast zwei Stunden vor Tagesanbruch -- die rechte Nachtzeit im
    Herbste, da, indem sogar der Schall zu schlummern scheint, die Stra�en
    schweigend und verlassen und die Schlemmer und Schwelger nach Hause
    getaumelt sind, um zu tr�umen -- als der Jude wachend in seiner alten
    H�hle mit einem so bleichen und verzerrten Gesichte und so roten und
    blutunterlaufenen Augen dasa�, da� er weniger einem Menschen als einem
    greulichen, vom Grabe feuchten und von einem b�sen Geiste gepeinigten
    Gespenste glich.

    Er kauerte, in eine zerlumpte Bettdecke geh�llt, an einem kalten Herde
    und hatte die Blicke auf ein dem Erl�schen nahes Licht gerichtet,
    das auf einem Tische neben ihm stand. Die rechte Hand hielt er, wie
    in Gedanken verloren, an die Lippen und kaute an seinen langen,
    schwarzen Fingern�geln, so da� man in dem sonst zahnlosen Munde
    einige Vorderz�hne erblickte, die einem Hunde oder einer Ratte h�tten
    angeh�ren k�nnen.

    Auf einer Matratze am Boden ausgestreckt lag Noah Claypole in festem
    Schlafe. Zwischen ihm und dem Lichte schweiften die zerstreuten
    Blicke des alten Mannes bisweilen hin und wieder, in dessen Innerem,
    einander dr�ngend, unruhige Gedanken und st�rmische Leidenschaften
    wogten und w�hlten -- bitterer Verdru� �ber das Mi�lingen seines
    gewinnverhei�enden Planes, t�dlicher Ha� gegen das M�dchen, das
    hinterlistig mit Fremden zu verkehren gewagt hatte, g�nzliches
    Mi�trauen in die Aufrichtigkeit ihrer Weigerung, ihn zu verraten,
    Ingrimm dar�ber, sich an Sikes nicht r�chen zu k�nnen, Furcht vor
    Entdeckung, Verurteilung und Tod, die wildeste, durch das alles
    entz�ndete Wut und neue Pl�ne der Arglist und schw�rzesten Bosheit. Er
    sa� da, ohne auch nur im mindesten seine Stellung zu ver�ndern oder
    anscheinend die Zeit zu beachten, bis der Schall von Fu�tritten auf der
    Stra�e bei seinem feinen Geh�r seine Aufmerksamkeit zu erregen schien.

    �Endlich,� murmelte er, �ber die trocknen, fieberhei�en Lippen mit der
    Hand hinfahrend, �endlich!�

    Die Glocke ert�nte leise, er ging hinaus und kehrte bald darauf mit
    einem Manne zur�ck, der bis an das Kinn vermummt war und ein B�ndel
    unter dem Arme trug. Es war Sikes.

    �Da�, sagte der verwegene Raubgesell, das B�ndel auf den Tisch werfend.
    �Mach draus, was du kannst. Es hat mir M�he genug gekostet; ich meinte
    schon vor drei Stunden hier sein zu k�nnen.�

    Fagin verschlo� das B�ndel, setzte sich wieder, blieb stumm, blickte
    jedoch nach Sikes scharf hin�ber, und seine Lippen zitterten so heftig,
    und sein Gesicht war infolge der in ihm w�hlenden Leidenschaften
    so ver�ndert, da� der Dieb unwillk�rlich sich zur�cklehnte und ihn
    best�rzt ansah.

    �Was gibt's?� fuhr er auf. �Zu allen Teufeln, was siehst du mich so an?�

    Der Jude hob die rechte Hand empor und sch�ttelte den bebenden
    Zeigefinger; allein seine Bewegung war so heftig, da� er kein Wort
    hervorzubringen imstande war.

    �Gott verdamm' mich!� rief Sikes, in seine Brusttasche greifend, aus.
    �Er ist verr�ckt geworden. Ich mu� auf meiner Hut gegen ihn sein.�

    �Nein, o nein�, brachte Fagin endlich hervor. �Ihr -- Ihr seid's nicht,
    Bill. Gegen Euch hab' ich nichts -- gar nichts, Bill.�

    �Hm, 's ist auch ein Gl�ck f�r einen von uns -- gleichviel f�r wen�,
    sagte Sikes, ein Pistol absichtlich hervorziehend und in eine andere
    Tasche steckend.

    �Ich hab' Euch zu sagen was, Bill,� fuhr der Jude n�her r�ckend fort,
    �was Euch noch mehr wird erz�rnen als mich.�

    �So?!� entgegnete Sikes mit einer ungl�ubigen Miene. �Wenn's aber wahr
    ist, so tu's Maul auf und mach' g'schwind, oder Nancy wird glauben, da�
    ich verloren w�r'.�

    �Das hat sie schon ausgemacht bei sich selbst bestimmt genug!�

    Sikes blickte ihn ungewi� an, streckte die m�chtige Faust nach ihm
    aus, sch�ttelte ihn und forderte ihn barsch und polternd auf, sich
    deutlicher zu erkl�ren.

    �Denkt Euch,� sagte der Jude mit vor Wut fast erstickter Stimme,
    �der Bursch da schliche nachts hinaus auf die Stra�en, kn�pfte
    an Einverst�ndnisse mit unsern schlimmsten Feinden, g�be ihnen
    Beschreibungen von uns und unsern verborgensten Schlupfwinkeln,
    verriete unsere geheimsten Pl�ne und Taten, setzte auch hinzu noch viel
    L�gen -- was dann, was dann?�

    Sikes erkl�rte unter einer furchtbaren Verw�nschung, er w�rde ihm
    in einem solchen Falle den Sch�del unter den eisernen N�geln seiner
    Stiefel zermalmen.

    �Aber wie, wenn ich's t�te!� schrie der Jude fast, �ich, der ich wei�
    so viel und so viele kann bringen an den Galgen.�

    �Wei� nicht,� erwiderte Sikes, bei dem blo�en Gedanken die Z�hne
    zusammenbei�end und erblassend. �Aber ich w�rd' im Kerker was tun, da�
    sie mich in Eisen schlagen m��ten, und stellten sie mich mit dir vor
    Gericht, w�rd' ich dir vor den Richtern und allen den Kopf einschlagen.
    Ich w�rd' ne solche Kraft haben,� murmelte er, den sehnigen Arm auf und
    nieder schwingend, �da� ich ihn dir zu Brei schlagen k�nnte, als wenn
    ein belad'ner Frachtwagen dr�berhingegangen w�r'.�

    �W�rdet Ihr tun das wirklich?�

    �Ob ich's wohl tun w�rd'! Stell mich auf die Probe.�

    �Wenn's aber getan h�tte Charley oder der Baldowerer oder Bet oder --�

    �Ist mir gleichviel wer�, unterbrach Sikes ungeduldig. �Ich w�rd' ihn
    bezahlen, m�cht's sein, wer wollte.�

    Fagin blickte ihn abermals scharf an, winkte ihm, zu schweigen,
    beugte sich �ber den Schl�fer und sch�ttelte denselben, w�hrend Sikes
    verwundert und erwartungsvoll, die H�nde auf die Knie stemmend, dasa�.

    �Bolter, Bolter! Der arme Junge�, sagte Fagin, mit einer Miene
    emporblickend, in welcher die Vorahnung einer teuflischen Freude sich
    ausdr�ckte. �Er wird m�d' -- m�de davon, da� er hat m�ssen wach sein
    ihretwegen so lange -- ihr hat nachschleichen m�ssen noch so sp�t,
    Bill.�

    �Was willst du damit sagen?� fragte Sikes, sich zur�cklehnend.

    Der Jude antwortete nicht, setzte seine Bem�hungen, Noah zu wecken,
    fort, und es war ihm endlich einigerma�en gelungen.

    �Erz�hl' noch einmal -- da� der es h�rt auch�, sagte er, nach Sikes
    hinzeigend.

    �Was soll ich erz�hlen?� fragte Noah, noch halb im Schlafe.

    �Das von -- Nancy�, antwortete der Jude, Sikes fest am Arme fassend,
    wie um ihn zu verhindern, fortzueilen, bevor er genug geh�rt h�tte, und
    fragte darauf dem schlaftrunkenen Noah mit einer Wut, deren er nur mit
    M�he Herr zu bleiben vermochte, alles ab, was der Lauscher erhorcht
    hatte.

    �Und was sagte sie,� fragte er endlich mit wutsch�umenden Lippen, �was
    sagte sie vom vorigen Sonntage?�

    �Der Herr fragte sie, warum sie nicht am vorigen Sonntage gekommen
    w�re,� antwortete Noah, in welchem eine Ahnung davon auftauchte, wer
    Sikes sein m�chte; �und sie sagte, weil sie gewaltsam zur�ckgehalten
    worden w�re von Bill, dem Manne, von dem sie ihnen schon gesagt h�tte.�

    �Was weiter von ihm?� rief der Jude. �Was sagte sie von ihm weiter?
    Sag' ihm das, sag' ihm das!�

    �Es w�re nicht leicht f�r sie,� fuhr Noah fort, �aus dem Hause zu
    kommen, ohn' da� er w��te, wohin sie ginge, und sie h�tt' ihm daher,
    als sie das erstemal zu der Dame gekommen w�re, 'nen Schlaftrunk
    eingeben m�ssen -- ha, ha, ha.�

    �H�ll' und Teufel!� schrie Sikes, von dem Juden sich losrei�end. �La�
    mich!�

    Er st�rzte w�tend hinaus, Fagin rief und eilte ihm nach, w�rde ihn
    jedoch nicht zur�ckgehalten haben, wenn die Haust�r nicht verschlossen
    gewesen w�re.

    �La� mich 'naus,� tobte er, �oder nimm dich in acht! La� mich 'naus --
    h�rst du?�

    �Ein Wort, Bill -- blo� ein einziges Wort�, versetzte der Jude, die
    Hand auf das T�rschlo� legend und mit verstellter Besorglichkeit: �Ihr
    -- Ihr wollt doch nicht tun etwas zu -- zu Gewaltsames, Bill?�

    Der Tag brach an, es war hell genug, als sie einander in das Gesicht
    schauten, um deutlich sehen zu k�nnen, und in ihren Augen blitzte ein
    Feuer, dessen Bedeutung nicht mi�zuverstehen war.

    �Ich meine�, setzte Fagin hinzu, einsehend, da� Verstellung nicht
    mehr m�glich war, �nichts Gewaltsames, wodurch wir geraten k�nnten in
    Gefahr. Fein listig, Bill, und seid nicht zu verwegen.�

    Er hatte unterdes aufgeschlossen, Sikes antwortete nicht, ri� die T�r
    auf, st�rzte hinaus und eilte, ohne rechts oder links zu schauen, ohne
    eine Gesichtsmuskel zu bewegen oder ein zorniges Wort zu murmeln,
    mit verbissenen Z�hnen und trotzig-blutd�rstiger Entschlossenheit
    nach seiner Wohnung. Er ging mit leisen Schritten hinauf, �ffnete und
    verschlo� die T�r seines Zimmers, stellte einen schweren Tisch gegen
    sie und schob den Bettvorhang zur�ck.

    Und da lag Nancy, halb angekleidet. Sie schreckte aus dem Schlaf empor.

    �Steh auf�, sagte er.

    �Bist du es?� rief sie ihm, erfreut �ber seine R�ckkehr, entgegen.

    �Ja. Steh auf!�

    Es brannte ein Licht -- er schleuderte es unter den Kaminrost. Sie
    stand auf und ging nach dem Fenster, um den Vorhang aufzuziehen.

    �La� das�, herrschte er ihr zu. �'s ist hell genug f�r das, was wir zu
    tun haben.�

    �Bill,� sagte sie best�rzt, �was seht Ihr mich so an?�

    Er heftete eine kurze Weile schnaubend und mit wogender Brust die
    Blicke auf sie, packte sie darauf beim Kopfe und der Kehle, zog sie in
    die Mitte des Gemachs, warf einen einzigen Blick nach der T�r und legte
    seine schwere Hand auf ihren Mund.

    �Bill, Bill!� keuchte sie, in Todesangst unter seinem Griff sich
    str�ubend, �ich will nicht schreien -- nicht weinen -- h�rt mich --
    sprecht doch nur -- sagt mir, was ich getan habe.�

    �Wei�t es selbst, du Satan in Dirnengestalt. Bist belauert gewesen
    gestern abend; ich wei� jedes Wort, was du gesagt hast.�

    �Oh, um der Liebe des Himmels willen,� rief sie, sich fest an ihn
    anklammernd, �dann schont mein Leben, wie ich Eures geschont habe.
    Bill, bester Bill, Ihr k�nnt mich ja nicht morden wollen. Bedenkt, was
    ich gestern abend um Euretwillen aufgegeben habe. Ihr sollt Zeit haben,
    es zu bedenken, Euch dies Verbrechen zu ersparen -- ich lasse Euch
    nicht los, nimmermehr! Bill, Bill, um Gottes Barmherzigkeit, um Euret-
    und um meinetwillen, besinnt Euch, eh' Ihr mein Blut vergie�t. Bei
    meiner s�ndigen Seele, ich bin Euch treu gewesen!�

    Er suchte sich gewaltsam von ihr loszumachen, allein vergebens, sie
    hielt mit der Kraft der Verzweiflung fest.

    �Bill,� rief sie und bem�hte sich, den Kopf auf seine Brust zu legen,
    �der Herr und die liebe Dame boten mir einen Zufluchtsort au�er Landes
    an. La�t mich noch einmal zu ihnen, da� ich sie auf den Knien anflehe,
    Euch dieselbe Liebe und G�te zu erweisen, und dann la�t uns aus dieser
    H�hle entfliehen und weit von hier ein besseres Leben anfangen und
    unser voriges Leben, ausgenommen im Gebet, vergessen und uns nie
    wiedersehen. Es ist zur Reue niemals zu sp�t. Sie sagten es mir -- ich
    f�hle es jetzt -- aber wir m�ssen Zeit -- ein wenig, ein wenig Zeit
    haben!�

    Er befreite einen seiner Arme und ergriff seine Pistole; doch so
    w�tend er war, der Gedanke, da� sogleich alles entdeckt werden w�rde,
    wenn er Feuer g�be, flog ihm durch den Sinn, und er schlug sie daher
    mit aller Kraft, die er zu sammeln vermochte, zweimal auf das zu ihm
    emporgehobene, das seinige fast ber�hrende Gesicht.

    Sie wankte und st�rzte, fast erblindet von dem aus einer tiefen Wunde
    in ihrer Stirn hervorstr�menden Blute, zu Boden, richtete sich jedoch
    m�hsam wieder auf die Knie, zog ein wei�es Tuch -- das ihr von Rose
    geschenkte -- aus dem Busen und hielt es in den gefalteten H�nden so
    hoch, als es ihre schwachen Kr�fte erlaubten, zum Himmel empor und
    flehte um Erbarmen zu ihrem Sch�pfer.

    Sie war gr��lich anzuschauen. Der M�rder wankte zur�ck nach der Wand,
    hielt die Hand vor die Augen, um sie nicht zu sehen, ergriff einen
    schweren Knotenstock und schlug sie nieder.




    48. Kapitel.

        Sikes' Flucht.


    In der ganzen gro�en Hauptstadt war an diesem Morgen sicher keine so
    greuliche, ruchlose Tat geschehen. Die Sonne -- die helle Sonne, die
    nicht blo� Licht, sondern neues Leben, Hoffnung und r�stige Frische den
    Menschen zur�ckbringt -- ging strahlend auf �ber der menschenerf�llten
    Stadt und ergo� ihren Glanz durch kostbar bemalte Scheiben wie durch
    papierverklebte Fenster und hinein in den himmelanstrebenden Dom wie
    in die schlechteste, niedrigste H�tte. Sie erhellte auch das Gemach,
    in welchem die ermordete Nancy lag. Sikes bem�hte sich, dem Eindringen
    ihres Lichtes zu wehren, jedoch vergeblich, und hatte das M�dchen beim
    ungewissen D�mmerscheine des Morgens einen f�rchterlichen Anblick
    dargeboten, so war ihre blutige Gestalt bei voller Tageshelle noch
    zehnmal unheimlicher und schauerlicher anzuschauen.

    Sikes war aus Furcht nicht von der Stelle gewichen. Er hatte ein leises
    �chzen der jammervoll Daliegenden vernommen, ein Zucken ihrer Hand
    gewahrt und aber- und abermals geschlagen, denn Schrecken und Angst
    waren bei ihm zu der Erbitterung des Hasses hinzugekommen. Er warf eine
    Decke �ber sie; doch es war noch f�rchterlicher, im Geiste ihre Augen
    zu schauen, nach ihm sich wenden und dann emporstarren zu sehen, als
    wenn sie des Himmels Rache herabriefen. Er entfernte die Decke wieder,
    und da lag der schreckliche Leichnam, aus dessen Wunden das Blut noch
    langsam hervorquoll.

    Er z�ndete Feuer an und steckte den Knotenstock hinein, an welchem
    Haare der Ermordeten klebten, die er, trotz seiner Eisenfestigkeit,
    mit Zagen von den Flammen ergreifen sah, und lie� ihn darin, bis er
    zerbrach und zu Asche verbrannte. Er wusch sich und rieb seine Kleider
    ab. Sie hatten Flecke, die nicht ausgehen wollten, und er schnitt die
    St�cke heraus und verbrannte sie. Das ganze Gemach war blutbefleckt --
    sogar die F��e des Hundes waren blutig.

    Er hatte w�hrend dieser ganzen Zeit nicht nach der Leiche
    zur�ckgesehen, nicht ein einziges Mal, und ging, den Hund mit sich
    fortziehend, ohne hinzublicken nach der T�r, verschlo� sie und verlie�
    das Haus. -- Er schritt quer �ber die Stra�e und schaute nach dem
    Fenster hinauf, um sich zu �berzeugen, da� von au�en nichts zu sehen
    w�re. Das Fenster war durch den Vorhang verh�llt, den sie aufziehen
    wollte, um dem Lichte freien Zugang zu verschaffen, das sie aber nie
    wiedersehen sollte. Ihre Leiche lag ganz in der N�he -- er wu�te es --
    und wie hell die Sonne das Fenster erleuchtete!

    Es war ihm jedoch Erleichterung, das Zimmer verlassen zu haben; er
    pfiff dem Hunde und entfernte sich eiligen Schrittes. Er ging durch
    Islington und �ber Highgate-Hill, ungewi�, wohin er sich wenden sollte,
    hatte endlich Hampstead hinter sich gelassen, befand sich im Freien,
    legte sich hinter eine Hecke, schlief ein, erwachte jedoch bald wieder
    und irrte von neuem umher, bald eilend, bald z�gernd, rastlos selbst,
    wenn er bisweilen rastete. In Hendon gedachte er irgendwo einzukehren,
    allein sogar die Kinder vor den T�ren schienen ihn argw�hnisch
    anzublicken, der Mut fehlte ihm, einen Trunk oder einen Bissen Brot
    zu fordern, und er suchte das Freie wieder auf, obwohl ihn die
    vielst�ndige Wanderung, die ihn immer und immer wieder auf denselben
    Fleck zur�ckf�hrte, fast g�nzlich ersch�pft hatte.

    Um neun Uhr abends wagte er sich endlich in ein kleines Gasthaus in
    Hatfield hinein. Im Schenkst�bchen am Feuer sa�en einige l�ndliche
    Arbeiter. Sie machten Platz f�r den unbekannten Gast, allein er setzte
    sich in den fernsten Winkel und a� und trank allein, seinem erm�deten
    Hund von Zeit zu Zeit ein St�ck zuwerfend. Die Arbeiter unterhielten
    sich von ganz gew�hnlichen Dingen, und er schlummerte schon ein, als
    l�rmend ein Mann eintrat, der halb Hausierer, halb Marktschreier zu
    sein schien und sogleich anfing, seine Waren ruhmrederisch und unter
    mannigfachen Scherzen, wie sie zu dem Orte sich schicken mochten,
    anzupreisen.

    �Diese K�gelchen hier�, sagte er in Erwiderung auf eine Frage eines der
    Arbeiter, �sind ein untr�gliches und unfehlbares Mittel, aus allerlei
    Art Zeug alle Arten von Flecken auszutilgen. Hat eine Dame ihre Ehre
    befleckt, so braucht sie nur ein solches K�gelchen zu genie�en. Will
    ein Herr seine Ehre beweisen, kann er's ebensogut mit 'nem solchen
    K�gelchen tun als mit 'ner Pistolenkugel und noch besser, denn der
    Geschmack ist viel schlechter. Wer kauft? Das St�ck 'nen Penny --
    oder auch zwei Halbpence oder vier Heller -- mir ist's ganz gleich.
    Sie gehen so rei�end ab, da� sie nur selten zu haben sind; vierzehn
    Wasserm�hlen, sechs Dampfmaschinen und eine galvanische Batterie sind
    unaufh�rlich in Arbeit und k�nnen nicht schnell genug fabrizieren, um
    die K�ufer zu befriedigen, obgleich die angestellten Arbeiter sich
    totarbeiten und die Witwen mit zwanzig Pfund j�hrlich f�r jedes Kind
    pensioniert werden und mit 'ner Pr�mie f�r Zwillinge. Alle Flecke gehn
    davon aus, Fettflecke, Wein- und Farbe- und Wasser- und Blutflecke.
    Schauen Sie hier! Da ist ein Fleck auf dem Hute 'nes Gentleman, den ich
    'runterbringen werde, eh' er mir 'nen Krug Ale bringen lassen kann.�

    �Wollt Ihr wohl meinen Hut liegen lassen!� rief Sikes emporschreckend.

    �Sir,� fuhr der Hausierer, den Arbeitern zublinzelnd, fort, �ich werde
    den Fleck 'runter haben, eh' Sie zu mir herkommen k�nnen. Gentlemen,
    Sie bemerken den dunkeln Fleck auf dem Hute des Gentleman, nicht gr��er
    als ein Schilling, aber dicker als eine halbe Krone. Gleichviel, ob's
    ein Fettfleck ist, oder ein Wein-, ein Farbe-, ein Wasser- oder ein
    Blutfleck --�

    Er kam nicht weiter, denn Sikes stie� mit einer schrecklichen
    Verw�nschung den Tisch um, entri� ihm den Hut, schritt w�tend
    aus dem Hause hinaus und wandte sich in derselben Verwirrung und
    Unentschlossenheit, die ihn, ihm selber zum Trotze, den ganzen Tag
    nicht hatte verlassen wollen, wieder nach der Stadt zur�ck. Vor dem
    Posthause stand eine Londoner Diligence, und sorgf�ltig den Schein
    ihrer Laternen meidend, n�herte er sich ahnungsvoll, um zu horchen.

    Er hatte eine Zeitlang dagestanden, als ein Wildw�rter zu dem
    Kondukteur trat, der am Fenster des Bureaus auf seine Abfertigung
    wartete, und ihn fragte, ob es nichts Neues g�be.

    �Das Korn ist ein bissel gestiegen�, lautete die Antwort. �Auch h�rte
    ich von 'ner Mordtat, begangen in der Gegend von Spitalsfield -- doch
    wer wei�? Es wird entsetzlich gelogen.�

    �Es ist vollkommen wahr�, nahm ein Reisender das Wort. �Es ist eine
    h�chst schauderhafte Mordtat gewesen.�

    �Ist sie denn an einem Manne oder an einer Frau begangen, Sir?�

    �An einem M�dchen, und man sagte --�

    Hier wurde der Kutscher ungeduldig und rief dem Kondukteur zu, da� er
    sich beeilen m�chte.

    �Komme schon,� rief der Kondukteur heraustretend zur�ck, �wie auch die
    reiche junge Dame schon kommt, die sich in mich verlieben wird, ich
    wei� nur nicht, wann.�

    Er stieg hinauf, stie� in sein Horn, und die Diligence rasselte fort.

    Sikes stand da, anscheinend unbewegt und nur zweifelhaft, wohin er sich
    wenden sollte. Endlich schlug er den Weg nach St. Albans ein.

    Als er die Stadt hinter sich hatte und sich in der Finsternis auf der
    einsamen Stra�e befand, bem�chtigte sich seiner eine Be�ngstigung,
    so furchtbar, wie wenn sie ihm das Herz abdr�cken wollte. Alles um
    ihn her, wirkliche Gegenst�nde wie Schatten, ob es sich regen mochte
    oder nicht, nahm eine schreckliche Gestalt an; allein noch unendlich
    f�rchterlicher war die greuliche der Erschlagenen, die ihm dicht auf
    den Fersen mit feierlichen, geisterhaften Schritten nachfolgte. Er sah
    sie deutlich in der Finsternis, h�rte ihre Kleider in den Bl�ttern
    rauschen, und jeder Windhauch f�hrte seinem Ohre ihr letztes leises
    �chzen zu. Stand er still, so tat sie es ihm nach; lief er, so folgte
    sie ihm auch -- nicht im Laufe, was ihm eine Herzenserleichterung
    gewesen sein w�rde -- sondern wie eine Leiche, begabt nur mit
    mechanischer Bewegungskraft und getragen von einem traurigen, langsam
    daherrauschenden und sich weder verst�rkenden noch abnehmenden
    Lufthauche.

    Mehreremal drehte er sich mit einem verzweifelten Entschlusse um,
    gewillt, das Phantom zu verscheuchen, und wenn es ihn mit seinen
    Blicken t�tete; aber dann stand ihm das Haar zu Berge und das
    Blut still, denn die Gestalt hatte sich mit ihm umgedreht und war
    fortw�hrend hinter ihm. Am Morgen war sie vor ihm hergegangen --
    jetzt folgte sie ihm. Er stellte sich mit dem R�cken an die Wand
    eines steilen Grabens und hatte das Gef�hl, da� sie, in deutlichen
    Umrissen gegen den kalten Nachthimmel abstechend, vor ihm stand.
    Er warf sich nieder auf die Stra�e, und sie stand ihm zu H�upten,
    aufgerichtet, stumm und regungslos, gleich einem lebendigen Grabsteine
    mit blutgeschriebener Inschrift.

    Sage niemand, da� M�rder der Gerechtigkeit entgingen oder da� die
    Vorsehung schlummere! Der M�rder Sikes erduldete in einer einzigen
    Minute die Angst und Pein eines gewaltsamen Todes hundertfach.

    Er erblickte einen Schuppen in einem Felde, welcher ein Obdach f�r die
    Nacht darbot. Vor der T�re desselben standen drei hohe Pappelb�ume, die
    das Innere noch finsterer machten, und der Wind s�uselte unheimlich in
    ihren Bl�ttern. Es war unm�glich, er konnte nicht bis zu Tagesanbruch
    fortwandern und streckte sich dicht an der Wand nieder -- um neuen
    Qualen zum Raube zu werden. Denn jetzt trat ein Gesicht vor ihn, noch
    schrecklich beharrlicher und grausiger als das, welchem er entronnen
    war. Zwei starre, halbge�ffnete Augen, glanzlos und gl�sern, erschienen
    ihm mitten in der Finsternis, hatten ihr eignes Licht, gaben aber keins
    von sich. Es waren ihrer nur zwei, aber sie waren �berall. Bedeckte
    er seine Augen, so stand sein Zimmer mit allem, was es enthielt, so
    deutlich vor ihm, wie wenn er sich darin bef�nde. Alles war an seinem
    Orte, auch die Leiche an dem ihrigen, und ihre Augen waren so gl�sern
    und starr wie in der Minute, als er hinausschlich. Er sprang auf und
    eilte wieder in das Freie. Die Gestalt war hinter ihm. Er ging in den
    Schuppen zur�ck und dr�ckte sich wieder dicht an die Wand, und die
    Augen waren wieder da, noch bevor er sich niedergelegt hatte.

    Er bebte an allen Gliedern, und kalter Angstschwei� bedeckte ihn von
    Kopf bis zu F��en, als pl�tzlich aus weiter Ferne verwirrtes Rufen und
    Schreien an sein Ohr drang. Es erschien ihm als eine Wohltat, eine
    wirkliche Ursache zu Furcht und Schrecken zu erhalten. Kraft und Mut
    kehrten ihm bei der Aussicht auf pers�nliche Gefahr zur�ck, er raffte
    sich auf, eilte hinaus und sah den Himmel weithin von einer furchtbaren
    Feuersbrunst ger�tet. Die Sturmglocke ert�nte, und lauter und lauter
    wurde der L�rm und das Get�se. Es war ihm, als w�re ein neues Leben
    in ihm erwacht. Er rannte, sein Hund wie toll vor ihm her, nach der
    Richtung hin, �ber Hecken und Gr�ben, Mauern und Tore, kein Hindernis
    achtend, und langte atemlos an.

    Es standen viele H�user in Flammen, und die Angst, das Ger�usch, die
    Verwirrung waren grenzenlos. Er schrie selbst mit, bis er heiser war,
    und st�rzte sich, um seinem Ged�chtnisse und sich selbst zu entfliehen,
    in den dichtesten Haufen, arbeitete bald an den Spritzen, erstieg bald
    auf wankenden Leitern die h�chsten Dachgiebel, war �berall und trotzte
    jeder Gefahr; allein er schien ein bezaubertes Leben zu haben und
    empfand, bis der Tag graute, keine Spur von Erm�dung, trug nicht die
    kleinste Brandwunde, keine Beule, keine Schramme davon.

    Als jedoch die wahnsinnige Aufregung vor�ber war, kehrte ihm mit
    zehnfacher Gewalt das schreckliche Bewu�tsein seines Verbrechens
    zur�ck. Er blickte argw�hnisch umher, denn die Leute standen hier
    und da beieinander und redeten untereinander, und er f�rchtete, der
    Gegenstand ihrer Gespr�che zu sein. Der Hund gehorchte seinem Winke,
    und beide stahlen sich davon. Die W�rter einer Spritze forderten ihn
    auf, ihren Morgenimbi� mit ihnen zu teilen. Er nahm ein St�ck Brot
    und einen Trunk Bier an. Sie waren aus London und fingen an, von der
    Mordtat zu sprechen. �Er ist nach Birmingham gegangen,� sagte einer
    von ihnen, �aber sie werden ihn bald fassen, denn die Polizei hat ihre
    Sp�her schon ausgeschickt, und bis morgen wird nur der eine Schrei im
    ganzen Lande sein. >Wo ist der M�rder?<�

    Er eilte fort und ging, solange die F��e ihn tragen wollten, warf
    sich an einem entlegenen Orte nieder, verfiel in einen langen, aber
    unruhigen, oft unterbrochenen Schlaf, setzte unentschlossen und
    ungewi�, ge�ngstet von der Furcht vor einer zweiten einsamen Nacht,
    seine Wanderung wieder fort und fa�te pl�tzlich den verzweifelten
    Entschlu�, nach London zur�ckzukehren.

    �Kann doch wenigstens dort mit jemand sprechen,� dachte er, �und hab'
    ein gutes Versteck. Sie suchen mich da am letzten. Ich halte mich
    'ne Woche still, zwinge Fagin, zu blechen und schiffe 'n�ber nach
    Frankreich. Gott verdamm mich, ich wag's!�

    Sein Plan war, nach Dunkelwerden auf Schleichwegen Fagins Wohnung zu
    erreichen. Aber der mit ihm vermi�te Hund konnte seine Entdeckung
    und Verhaftung veranlassen. Er beschlo�, ihn zu ers�ufen, hob einen
    schweren Stein auf und kn�pfte denselben in sein Taschentuch. Ein
    Gew�sser war in der N�he, er lockte den Hund, allein lange mit
    vergeblicher M�he; die Blicke seines brutalen Herrn mochten den
    Instinkt des Tieres noch versch�rft haben. Sikes schmeichelte und
    drohte, der Hund kroch endlich zu ihm heran, sprang aber, als er sich
    pl�tzlich gefa�t f�hlte, zur�ck, lief davon, und Sikes mu�te seine
    Wanderung allein fortsetzen.




    49. Kapitel.

        Die endlich stattfindende Unterredung zwischen Monks und Mr.
        Brownlow.


    Es wurde dunkel, als Mr. Brownlow mit zwei M�nnern aus einem Mietswagen
    stieg, der vor seinem Hause hielt; die letzteren halfen einem dritten
    Manne heraus und dr�ngten ihn rasch durch die ge�ffnete T�r hinein. Der
    Mann war Monks.

    Brownlow ging ihnen schweigend in ein hinteres Zimmer voran. Vor
    der T�r desselben stand Monks widerstrebend still, und die beiden
    handfesten M�nner sahen Brownlow fragend an.

    �Entweder oder�, sagte Brownlow. �Die Folgen des einen wie des andern
    sind ihm bekannt. Weigert er sich, hineinzugehen, so f�hrt ihn aus dem
    Hause, ruft die Polizei zu Hilfe und klagt ihn in meinem Namen als
    Kapitalverbrecher an.�

    �Wie k�nnen Sie sich unterstehen, mich so zu nennen?� fuhr Monks auf.

    �Wie k�nnen Sie es wagen, mich zu einer Anklage gegen Sie zu dr�ngen,
    junger Mensch?� entgegnete Brownlow, ihn sehr bestimmt anblickend.
    �Werden Sie unsinnig genug sein, mein Haus zu verlassen? La�t ihn los!
    So, Sir. Sie k�nnen jetzt gehen -- und wir k�nnen Ihnen nachfolgen.
    Aber ich gebe Ihnen mein Wort darauf, sobald Sie den Fu� vor die T�r
    setzen, sind Sie auch schon wegen Betruges und Raubes verhaftet. Ich
    bin fest entschlossen. Sind Sie es auch -- nun wohl! -- aber Ihr Blut
    kommt auf Ihr eigenes Haupt.�

    �Aus wessen Macht bin ich auf offener Stra�e aufgegriffen und von
    diesen Schuften hierher gebracht worden?�

    �Ich verantworte, was die Leute getan haben. Beklagen Sie sich �ber
    Freiheitsberaubung -- es stand in Ihrer Gewalt, ihr auf dem Wege
    hierher ein Ende zu machen; Sie erachteten es aber selbst f�r r�tlich,
    sich ruhig zu verhalten. Wollen Sie die Gesetze anrufen -- tun Sie's;
    allein ich werde es gleichfalls tun und Ihnen keine Milde mehr zeigen,
    mich nicht bem�hen, Sie zu retten, wenn die Sachen erst einmal vor den
    Richter gekommen sind.�

    Monks war offenbar unentschlossen geworden.

    �Entschlie�en Sie sich rasch�, fuhr Brownlow mit ruhiger Festigkeit
    fort. �Wollen Sie, da� ich Anklagen gegen Sie vorbringe, deren Ausgang
    ich schaudernd vorhersehe, so wissen Sie, was Sie zu tun haben;
    w�nschen Sie Nachsicht und Vergebung von mir und den von Ihnen schwer
    Gekr�nkten, so treten Sie hinein und nehmen Sie, ohne ein Wort zu
    sagen, dort auf jenem Stuhle Platz, der schon zwei Tage auf Sie
    gewartet hat.�

    Monks z�gerte noch einige Augenblicke, ging indes endlich hinein und
    setzte sich. Brownlow befahl den beiden M�nnern, die T�r zu verriegeln
    und wieder zur Stelle zu sein, wenn er klingelte.

    �Eine saubere Behandlung, die ich von dem �ltesten Freunde meines
    Vaters erfahre�, sagte Monks, Hut und Mantel ablegend.

    �Gerade weil ich Ihres Vaters �ltester Freund bin, junger Mann,�
    erwiderte Brownlow, �weil einst die Hoffnungen und W�nsche meiner
    gl�cklichen Jugendzeit an ihn sich ankn�pften und an ein holdes Wesen
    von seinem Blute, das in seinen jungen Jahren zu Gott zur�ckkehrte und
    mich einsam und allein hier zur�cklie�; -- weil er, noch ein Knabe,
    mit mir an seiner einzigen Schwester Sterbebette kniete, an dem Morgen
    kniete, der sie, wenn es der Himmel nicht anders gewollt h�tte, zu
    meinem Weibe gemacht haben w�rde: -- weil mein wundes Herz von der
    Zeit an bis zu seinem Tode bei all seinen Pr�fungen und Irrt�mern an
    ihm hing; -- weil alle teure Erinnerungen an ihn mein Herz erf�llen
    und selbst durch Ihren Anblick erneuert werden; -- das alles ist es,
    weshalb ich Sie jetzt nachsichtig behandle -- ja, Eduard Leeford, sogar
    jetzt -- Sie, der Sie err�ten m�ssen, dieses Namens so unw�rdig zu
    sein.�

    �Was hat der mit der Sache zu schaffen?� fragte Monks, der verstockt
    und stumm-verwundert die Bewegung des alten Herrn gewahrt hatte. �Was
    gibt mir der Name?�

    �Freilich gilt er Ihnen nichts�, versetzte Brownlow. �Er war aber
    der Ihrige, und noch gl�ht und erhebt mir altem Manne in so weiter
    Zeitenferne das Herz wie sonst, wenn ich ihn nur von fremden Lippen
    nennen h�re. Ich bin sehr, sehr erfreut, da� Sie ihn mit einem anderen
    vertauscht haben.�

    �Das klingt alles gar pr�chtig�, sagte Monks nach einem langen
    Stillschweigen, w�hrend dessen er sich trotzig hin und her gewiegt und
    Brownlow, die Augen mit der Hand bedeckend, dagesessen hatte. �Aber was
    wollen Sie von mir?�

    �Sie haben einen Bruder, dessen Name, in Ihr Ohr gefl�stert, als ich
    auf der Stra�e hinter Ihnen ging, fast allein schon gen�gte, Sie zu
    veranlassen, erstaunt und erschreckt mich hierher zu begleiten.�

    �Ich habe keinen Bruder. Sie wissen, da� ich das einzige Kind war,
    wissen es ebensogut wie ich selbst.�

    �H�ren Sie, was ich wei�, und Sie werden schon anders reden, schon
    aufmerken auf das, was ich Ihnen sage. Ich wei� allerdings, da� Sie der
    einzige und h�chst unnat�rliche Spr��ling aus der unseligen Verbindung
    waren, zu welcher elender Familienstolz Ihren ungl�cklichen Vater fast
    noch als Knaben n�tigte.�

    �Es gilt mir gleich, wie harte Ausdr�cke Sie w�hlen m�gen�, unterbrach
    ihn Monks mit einem h�hnischen Lachen. �Sie sind mit der Sache bekannt,
    und das ist mir genug.�

    �Mir sind aber auch das Elend und die langen Qualen bekannt,� fuhr
    Brownlow fort, �welche der unpassenden Verbindung folgten. Ich wei�,
    unter welcher Pein das ungl�ckliche Paar seine Kette durch die ihm
    verg�llte Welt nachschleppte; wei�, da� auf f�rmliche Gleichg�ltigkeit
    Beleidigungen, Widerwille, Ha� und Abscheu folgten, bis sie sich
    endlich trennten, um fern voneinander zu leben und in anderen Kreisen
    die lange Qu�lerei zu vergessen. Und Ihre Mutter verga� sie bald, Ihren
    Vater aber dr�ckte sie noch jahrelang zu Boden.�

    �Was weiter, als sie sich voneinander getrennt hatten?�

    �Die zehn Jahre �ltere Gattin verga� unter Zerstreuungen auf dem
    Festlande den jugendlichen Gatten, der daheim sein geknicktes Leben
    vertrauerte, bis er eine Verbindung mit neuen Freunden ankn�pfte -- und
    zum wenigsten dieser Umstand ist zu Ihrer Kenntnis gelangt.�

    �Nein�, sagte Monks, die Blicke wegwendend und auf den Boden stampfend,
    wie jemand, der alles abzuleugnen entschlossen ist. �Nein!�

    �Ihr Benehmen und Ihre Handlungen �berzeugen mich, da� Sie ihn nie
    vergessen, nie aufgeh�rt haben, mit Bitterkeit daran zur�ckzudenken.
    Ich rede von der Zeit vor f�nfzehn Jahren, wo Sie erst elf Jahre alt
    waren, und Ihr Vater nur einunddrei�ig z�hlte, -- denn, wie gesagt, er
    war fast noch ein Knabe, als ihn sein Vater zu heiraten zwang. Mu� ich
    Dinge erw�hnen, die einen Schatten auf das Andenken Ihres Erzeugers
    werfen, oder wollen Sie es mir ersparen und mir die Wahrheit enth�llen?�

    �Ich habe nichts zu enth�llen!� erwiderte Monks in offenbarer
    Verwirrung. �Reden Sie weiter, wenn Sie es nicht lassen k�nnen.�

    �Nun wohl�, sagte Brownlow. �Die neuen Freunde waren zun�chst ein
    Flottenoffizier, der sich aus dem aktiven Dienste zur�ckgezogen, und
    dessen Frau vor einem halben Jahre gestorben war. Sie hatten mehrere
    Kinder gehabt, von denen nur zwei die Mutter �berlebten, beide T�chter,
    die eine sch�n und neunzehn, die andere ein Kind, zwei bis drei Jahre
    alt.�

    �Was geht das mich an?� fragte Monks.

    �Sie wohnten�, fuhr Brownlow, anscheinend ohne die Unterbrechung zu
    beachten, fort, �in einer Grafschaft, in welche Ihren Vater seine
    Streifereien gef�hrt, und wo auch er seinen Wohnsitz aufgeschlagen
    hatte. Bekanntschaft, vertraulicher Umgang und Freundschaft folgten
    schnell aufeinander. Ihr Vater war begabt, wie es wenige M�nner sind
    -- er hatte seiner Schwester Z�ge und Seele. In dem Ma�e, wie der alte
    Offizier ihn kennen lernte, begann er, ihn wahrhaft zu lieben. Da� es
    dabei geblieben w�re! Doch seine Tochter tat dasselbe.�

    Der alte Herr hielt inne, sprach aber bald weiter, da er sah, da� sich
    Monks in die Lippen bi� und die Blicke an den Boden heftete.

    �Sie war am Schlusse des Jahres mit Ihrem Vater verlobt, feierlich
    verlobt, Ihr Vater der Gegenstand der ersten, treuinnigen, gl�henden,
    einzigen Liebe eines arglosen, unerfahrenen M�dchens.�

    �Ihre Erz�hlung wird lang�, bemerkte Monks, unruhig hin und her r�ckend.

    �Sie ist eine wahre und traurige,� versetzte Brownlow, �und Geschichten
    dieser Art pflegen lang zu sein; wenn sie von ungetr�btem Gl�cke
    handelte, so w�re sie ohne Zweifel sehr kurz. -- Endlich starb einer
    der reichen Verwandten, dessen Einflu� zu verst�rken Ihr Vater
    von Ihrem Gro�vater geopfert worden war, und hinterlie� ihm seine
    Panazee f�r alles Wehe -- Geld. Er mu�te nach Rom eilen, wo der
    Erblasser gestorben war und seine Angelegenheiten in gro�er Verwirrung
    hinterlassen hatte, erkrankte selbst am Tage nach seiner Ankunft und
    starb nach einiger Zeit, ohne f�r eine letztwillige Verf�gung Sorge
    getragen zu haben, so da� sein ganzes Verm�gen Ihrer Mutter und Ihnen
    zufiel, die mit Ihnen nach Rom eilte, sobald sie in Paris die Kunde
    seines Todes erhalten.�

    Monks hielt hier den Atem an und h�rte Brownlow in gro�er Spannung zu,
    obgleich er die Blicke nicht nach ihm hinwandte. Als Brownlow schwieg,
    ver�nderte er seine Stellung, wie wenn er sich pl�tzlich erleichtert
    f�hlte, und fuhr mit dem Tuch �ber sein gl�hendes Antlitz. Brownlow
    sprach langsam und die Blicke fest auf ihn heftend, weiter: �Bevor er
    au�er Landes ging und als er London ber�hrte, kam er zu mir.�

    �Davon habe ich nie geh�rt�, unterbrach Monks in einem Tone,
    der Unglauben ausdr�cken sollte, doch mehr auf eine unangenehme
    �berraschung hindeutete.

    �Er kam zu mir und lie� unter anderen Gegenst�nden ein von ihm selbst
    gemaltes Bild des ungl�cklichen M�dchens, seiner Verlobten, zur�ck,
    das er in anderen H�nden nicht zu lassen w�nschte, auf seiner eiligen
    Reise aber nicht wohl mitnehmen konnte. Er war fast zu einem Schatten
    zusammengeschwunden, sprach verst�rt von begangenem Ehrenraube und
    Verderben, das er angerichtet, und k�ndigte mir an, da� er entschlossen
    w�re, sein ganzes Verm�gen, ob auch mit gro�em Verluste, in bares
    Geld umzuwandeln, Ihrer Mutter und Ihnen einen Teil seiner neu zu
    erwerbenden Erbschaft auszusetzen, und das Land zu verlassen -- ich
    wu�te nur zu wohl, da� er nicht allein gehen w�rde --, um es nie
    wiederzusehen. Mehr bekannte er sogar mir, seinem alten Jugendfreunde,
    nicht, dessen starke Zuneigung in der Erde wurzelte, die sie bedeckte,
    die beiden so teuer gewesen war. Er versprach mir, zu schreiben und
    mir alles zu sagen, und mich dann noch ein -- das letztemal hienieden,
    wiederzusehen. Ach, es war schon das letztemal. Ich bekam keinen Brief
    und sah ihn nicht wieder. Als ich vernahm, da� er tot war, begab
    ich mich nach dem Schauplatz seiner -- wie die Welt es nennen w�rde
    -- s�ndlichen Liebe, um, wenn ich meine Bef�rchtungen wahr geworden
    f�nde, dem verirrten M�dchen ein mitleidiges Herz und eine Zuflucht
    anzubieten. Allein die Familie hatte vor einer kurzen Zeit die
    Angelegenheiten geordnet und war bei Nacht abgereist, niemand wu�te
    wohin.�

    Monks atmete freier und blickte mit einem triumphierenden L�cheln
    umher. Brownlow r�ckte seinen Stuhl n�her zu ihm und sagte: �Als Ihr
    Bruder -- ein verlorener, schwacher, in Lumpen geh�llter Knabe -- nicht
    durch Zufall, sondern durch eine h�here F�gung in meinen Weg geworfen
    und von mir gerettet wurde --�

    �Wie?!� rief Monks in heftiger Spannung aus.

    �Von mir�, wiederholte Brownlow. �Ich sagte es Ihnen, da� Sie schon
    aufmerken w�rden auf das, was ich Ihnen zu sagen ged�chte. Ja, von
    mir -- ich sehe, Ihr schlauer Verb�ndeter hat Ihnen meinen Namen
    verschwiegen, obwohl er nicht annehmen konnte, da� Ihnen derselbe
    bekannt w�re. W�hrend sich Ihr Bruder als Kranker und Wiedergenesener
    in meinem Hause befand, erkannte ich zu meinem lebhaften Erstaunen
    seine gro�e �hnlichkeit mit dem erw�hnten Bilde. Schon im ersten
    Augenblicke, als ich ihn sah, erinnerten mich seine Z�ge an einen
    alten Freund, nur da� mir alles unbestimmt blieb, wie wenn ich mir
    Traumbilder vergeblich deutlich und klar vor die Seele zur�ckzurufen
    suchte. Ich brauche Ihnen nicht zu sagen, da� er mir entf�hrt wurde,
    ehe ich seine Geschichte kennen lernte --�

    �Warum nicht?� fragte Monks hastig.

    �Sie wissen es sehr gut.�

    �Ich?�

    �Sie leugnen vergeblich und sollen bald sehen, da� ich noch mehr wei�.�

    �Sie -- Sie k�nnen nichts gegen mich beweisen�, stotterte Monks. �Tun
    Sie's, wenn Sie's imstande sind.�

    �Wir werden sehen�, sagte der alte Herr mit einem durchdringenden
    Blicke. �Der Knabe wurde mir entf�hrt, und meine Bem�hungen, ihn
    wieder aufzufinden, waren vergeblich. Da Ihre Mutter tot war, so
    konnten Sie allein, wenn irgend jemand, das Geheimnis enth�llen, und
    da Sie, wie ich geh�rt hatte, auf Ihrer Pflanzung in Westindien sich
    aufhielten -- wohin Sie nach Ihrer Mutter Tode gegangen waren, um den
    Folgen Ihres ruchlosen Lebenswandels hier in England zu entgehen --,
    so reiste ich Ihnen nach. Sie hatten sich unterdes wieder entfernt,
    und man glaubte, da� Sie sich in London bef�nden, doch vermochte
    niemand genauere Nachweisungen zu geben. Ich kehrte zur�ck. Ihren
    Gesch�ftsf�hrern war Ihr Wohnort vollkommen unbekannt; sie sagten,
    da� Sie ebenso geheimnisvoll k�men und gingen, wie Sie es immer getan
    h�tten, bisweilen wochen- und monatelang nicht erschienen und aller
    Wahrscheinlichkeit nach mit den schandbaren Menschen sich umhertrieben,
    denen Sie sich zugesellt, seit Sie ein trotziger, unlenksamer Knabe
    waren. Ich h�rte nicht auf, sie zu befragen, in T�tigkeit zu erhalten.
    Ich durchwanderte die Stra�en bei Nacht wie bei Tage, allein meine M�he
    war bis vor zwei Stunden fruchtlos, wo ich Ihrer endlich zum ersten
    Male ansichtig wurde.�

    �Und da Sie mich nun aufgefunden haben,� nahm Monks, sich dreist
    erhebend, das Wort, �was mehr? Betrug und Raub sind vollt�nende Worte
    -- und gerechtfertigt, wie Ihnen scheint, durch die eingebildete
    �hnlichkeit eines kleinen Landstreichers mit der Pinselei eines l�ngst
    Verstorbenen. Allein Sie wissen nicht einmal, ob aus der Verbindung des
    letzteren mit dem erw�hnten M�dchen ein Kind entspro� -- wissen das
    nicht einmal!�

    �Ich wei� es wirklich erst seit vierzehn Tagen�, erwiderte Brownlow,
    gleichfalls aufstehend. �Sie haben einen Bruder, wissen es und kennen
    ihn. Es war ein Testament vorhanden, das Ihre Mutter vernichtete, die
    Ihnen bei ihrem Tode das Geheimnis und den s�ndigen Gewinn hinterlie�.
    Das Testament nahm Bezug auf ein Kind, das Ihrem Vater noch geboren
    werden m�chte; es wurde geboren, Sie trafen mit ihm zusammen, und
    seine �hnlichkeit mit Ihrem Vater erweckte b�se Ahnungen in Ihnen. Sie
    suchten seinen Geburtsort auf, wo Beweise, lange unterdr�ckte Beweise
    seiner Geburt und Herkunft vorhanden waren. Sie vernichteten sie, und,
    wie Sie Ihrem j�dischen Schand- und Schuldgenossen sagten, sie liegen
    jetzt auf dem Grunde des Stromes, und die alte Hexe, die sie seiner
    Mutter nahm, fault in ihrem Sarge. Unw�rdiger Sohn, L�gner, Feigling,
    der du nachts mit R�ubern und M�rdern in finsteren Gem�chern verkehrst,
    -- der du durch sch�ndliche List an dem kl�glichen Tode einer
    Ungl�cklichen schuld bist, deren Wert Millionen von deinesgleichen
    aufwog, -- der du von deiner Wiege an dem Herzen deines Vaters nur
    Bitterkeit und Galle warst, -- du, in dessen angefaultem Innern die
    schlechtesten Leidenschaften so lange eiterten, bis sie einen Ausbruch
    in der scheu�lichen Krankheit fanden, die dein Antlitz zu einem Spiegel
    deiner teuflischen Seele gemacht hat, -- Eduard Leeford, setzt du mir
    auch jetzt noch Trotz entgegen?�

    �Nein, nein, nein!� st�hnte der durch so geh�ufte Beschuldigungen
    �berw�ltigte Feigling.

    �Jedes Wort,� rief Brownlow aus, �jedes Wort, das zwischen dir und dem
    �ber alles sch�ndlichen B�sewicht gewechselt worden, ist mir bekannt.
    Schatten an der Wand haben dein Gefl�ster vernommen und meinem Ohr
    zugef�hrt; der Anblick des unschuldigen, verfolgten Kindes hat selbst
    das Laster ergriffen und ihm den Mut und fast die Wesenheit der Tugend
    verliehen. Ein Mord ist begangen, an welchem du mindestens moralisch
    teilgenommen hast!�

    �Nein, nein�, unterbrach Monks. �Ich -- ich wei� nichts davon. Ich ging
    eben, um zu erfahren, was Wahres an der Sache w�re, als Sie mich mit
    sich fortf�hrten. Die Veranlassung der Tat war mir unbekannt -- ich
    glaubte, sie w�re nur durch einen gew�hnlichen Streit gegeben worden.�

    �Sie war keine andere als die teilweise Enth�llung Ihrer Geheimnisse�,
    sagte Brownlow. �Wollen Sie dieselben jetzt ganz offenbaren?�

    �Ja, ich will's!�

    �Alles vor Zeugen wiederholen und eine wahrhafte Aufzeichnung durch
    Ihre Namensunterschrift beglaubigen?�

    �Auch das verspreche ich.�

    �Ruhig in meiner Wohnung verweilen, bis es geschehen ist, und sich
    mit mir an den Ort begeben, den ich f�r den geeignetsten halte, dem
    Dokumente die vollkommenste G�ltigkeit zu verschaffen?�

    �Wenn Sie darauf bestehen, will ich auch das tun.�

    �Sie m�ssen noch mehr tun, dem guten, unschuldigen Kinde Ersatz
    leisten. Sie haben die Verf�gungen Ihres v�terlichen Testaments nicht
    vergessen. Bringen Sie dieselben, soweit sie Ihren Bruder betreffen,
    zur Ausf�hrung und gehen Sie dann, wohin es Ihnen beliebt. Sie d�rfen
    einander in dieser Welt nicht mehr begegnen.�

    W�hrend Monks auf und ab ging und der Aufforderung Brownlows und
    listigen Ausfl�chten, zwischen Furcht und Ha� schwankend, nachsann,
    wurde pl�tzlich die T�r aufgeschlossen und ge�ffnet, und herein trat in
    heftiger Aufregung Mr. Losberne.

    �Er wird ergriffen, wird heute abend ergriffen werden!� rief er.

    �Der M�rder?� fragte Brownlow.

    �Ja, ja. Sein Hund hat die Polizei auf die Spur gef�hrt. Sein
    Schlupfwinkel ist von allen Seiten umstellt, und die Beh�rden haben
    hundert Pfund ausgesetzt.�

    �Ich lege noch f�nfzig zu und will es auf der Stelle mit meinen eigenen
    Lippen verk�nden. Wo ist Maylie?�

    �Harry -- er warf sich, sobald er Sie mit dem jungen Menschen im
    Mietswagen sah, zu Pferde und sprengte fort, um sich den Verfolgern des
    M�rders anzuschlie�en.�

    �H�rten Sie nichts von dem Juden?�

    �Er wird in diesem Augenblick bereits festgenommen sein.�

    �Haben Sie Ihren Entschlu� gefa�t?� fragte Brownlow Monks leise.

    �Ja. Sie -- Sie werden mich nicht ausliefern?�

    �Nein. Aber Sie bleiben hier, bis ich zur�ckkehre. Ihre Sicherheit
    h�ngt einzig davon ab.�

    Brownlow und Losberne entfernten sich, und die T�r wurde wieder
    verschlossen.

    �Was haben Sie ausgerichtet?� fragte Losberne fl�sternd.

    �Soviel ich hoffen konnte und mehr. Veranstalten Sie auf �bermorgen
    abend die Zusammenkunft. Wir werden ein paar Stunden fr�her da sein,
    aber Ruhe bed�rfen, besonders die junge Dame, die vielleicht gr��erer
    Festigkeit ben�tigt sein m�chte, als Sie und ich jetzt voraussehen
    k�nnen. Doch mir kocht das Blut in den Adern, das arme ermordete
    Gesch�pf zu r�chen. Wohin mu� ich meine Schritte richten?�

    �Eilen Sie nur zuv�rderst nach dem Polizeiamte; ich will hierbleiben.�

    Die Herren nahmen hastigen Abschied voneinander, beide in einem
    unbez�hmbaren Fieber von Aufregung.




    50. Kapitel.

        Verfolgung und Entkommen.


    Unweit der Stelle des Themseufers, wo die Kirche von Rotherhithe steht
    und die Geb�ude am erb�rmlichsten und die Fahrzeuge auf dem Strome
    am schw�rzesten aussehen vom Kohlenstaube und dem Rauche der eng
    aneinander gebauten niedrigen H�user, befindet sich heutigestags die
    schmutzigste, widerw�rtigste und unheimlichste der vielen in London
    versteckten und der gro�en Mehrzahl der Bewohner der Hauptstadt selbst
    dem Namen nach unbekannten �rtlichkeiten.

    Um zu ihr zu gelangen, mu� man sich durch ein Labyrinth von kotigen
    und engen Stra�en hindurchwinden, die von den rohesten und �rmsten
    Uferbewohnern erf�llt und ihrem Verkehre gewidmet sind. In den L�den
    schaut man die wohlfeilsten und uneinladendsten Nahrungsmittel, an
    den Fenstern und T�ren der Altkleiderh�ndler die verschiedenartigsten
    Lumpen. Man arbeitet und st��t sich m�hsam weiter durch das
    Gedr�nge unbesch�ftigter Menschen der niedrigsten Klasse, Last- und
    Kohlentr�ger, frecher Weiber, zerlumpter Kinder, des recht eigentlichen
    Themseabschaums, indem ekelhafte Gegenst�nde und D�fte in allen
    Richtungen das Auge und den Geruchsinn beleidigen und das Ohr durch
    verwirrtes Ger�usch aller Art bet�ubt wird. Gelangt man endlich in die
    noch entlegneren, minder besuchten Winkelgassen, so scheinen wankende
    H�user zu beiden Seiten mit augenscheinlichem Einsturze zu drohen,
    und man sieht, wohin man blickt, halb eingefallene Schornsteine,
    erblindete oder zerschlagene Fenster und was nur sonst an Armut und
    Vernachl�ssigung erinnern mag.

    In einer solchen Umgebung, jenseits Dockhead im Borough Southwark,
    befindet sich die Jakobsinsel, umgeben von einem Sumpfgraben von sechs
    bis acht Fu� Tiefe und f�nfzehn bis zwanzig Fu� Breite zur Flutzeit,
    vormals der M�hlgraben genannt, jetzt bekannt unter dem Namen Folly
    Ditch. Sie ist eine Art Strombucht und kann bei hohem Wasser durch
    �ffnung der Schleusen bei den Leadm�hlen, von welchen sie ihre alte
    Benennung hat, ganz unter Wasser gesetzt werden. Steht man, wenn dies
    geschieht, auf einer der h�lzernen Br�cken, die bei der M�hlengasse
    �ber sie hin�berf�hren, so kann man sehen, wie die Bewohner der H�user
    zu beiden Seiten an den Hintert�ren und Fenstern Eimer und K�chenger�t
    aller Art herunterlassen, um Wasser zu sch�pfen, und erblickt h�lzerne
    Galerien, welche ein halbes Dutzend Hinterh�user verbinden, mit
    L�chern, aus denen sich auf die Lache hinunterschauen l��t; verklebte
    und verstopfte Fenster, aus welchen Stangen hervorstehen zum Trocknen
    von W�sche, die nicht vorhanden ist; die denkbar engsten, dumpfigsten,
    finstersten Gem�cher; halbversunkene, mi�farbige W�nde und zahllose
    �hnliche Anzeichen des Verfalls und Elends.

    Die Warenh�user der Jakobsinsel stehen leer und haben weder D�cher
    noch Fenster noch T�ren. Dem lebhaften Verkehre, der hier vor einigen
    Jahrzehnten stattfand, ist Ver�dung gefolgt. Die H�user haben keine
    Eigent�mer, stehen unbewohnt oder werden erbrochen und bewohnt von
    Leuten, die den Mut dazu und sonst keine Wohnst�tte haben, bei welchen
    entweder starke Beweggr�nde obwalten, in tiefer Verborgenheit zu
    leben, oder die sich in der allerbed�rftigsten und jammervollsten Lage
    befinden.

    In einem oberen Gemache eines dieser H�user, das etwas abgesondert
    stand, in anderen Beziehungen zu den verfallensten geh�rte, aber stark
    befestigte T�ren und Fenster hatte, von welchen die hinteren auf das
    beschriebene sumpfige Gew�sser hinausgingen, sa�en drei M�nner in
    tiefem d�steren Stillschweigen, einander von Zeit zu Zeit Blicke der
    Bangigkeit und angstvollen Erwartung zuwerfend. Es waren Toby Crackit,
    Tom Chitling und ein Raubgeselle von etwa f�nfzig Jahren, dem einst die
    Nase fast plattgeschlagen worden, und dessen Gesicht eine grauenvolle
    Narbe hatte, ohne Zweifel gleichfalls infolge einer Schl�gerei. Er war
    ein zur�ckgekehrter Deportierter und hie� Kags.

    �Ich wollte,� nahm endlich Toby, zu Chitling sich wendend, das Wort,
    �da� Ihr Euch ein anderes Bayes ausgesucht h�ttet, als Euch die beiden
    alten zu warm wurden, und nicht hierher gekommen w�ret.�

    �Freilich�, stimmte Kags bei; �warum tat'st das nicht, Dummkopf?�

    �Ich glaubte, Ihr w�rdet etwas vergn�gter gewesen sein, mich zu sehen�,
    antwortete Chitling mit tr�bseliger Miene.

    �Ja seht, junger Herr,� sagte Toby, �wenn sich einer so exklusiv h�lt,
    wie ich's getan habe, und somit in 'nem gem�tlichen Hause sitzt,
    da niemand reinguckt und das niemand umschn�ffelt, so ist's ein
    verfluchtes Ding, die Ehre 'nes Besuchs von 'nem jungen Gentleman in
    Eurer Lage zu haben, so respektabel und angenehm es sonst sein mag,
    nach Umst�nden Karten mit ihm zu spielen.�

    �Besonders,� f�gte Kags hinzu, �wenn der exklusive junge Gentleman
    'nen Freund bei sich hat, der aus fremden L�ndern eher zur�ckgekehrt
    ist, als er erwartet wurde, und zu viel Bescheidenheit besitzt, um zu
    w�nschen, nach seiner Heimkehr den Richtern vorgestellt zu werden.�

    Toby Crackit schwieg eine Zeitlang und fragte darauf Chitling, doch
    nicht mehr in seinem leichtfertig renommistischen Tone, wann Fagin
    ergriffen worden w�re.

    �Heute nachmittag um zwei Uhr�, erwiderte Tom. �Charley und ich
    entkamen durch den Waschhausschornstein, und Bolter plumpste mit dem
    Kopfe zuerst in 'ne leere Wassertonne hinein; aber seine langen Beine
    standen heraus, und er wurde auch gefa�t.�

    �Und Bet?�

    �Sie ging, um die Leiche anzusehen, und fing an zu toben und zu rasen
    bei dem Anblicke und wollte sich den Kopf einrennen. Sie legten ihr
    drum 'ne Zwangsjacke an und brachten sie ins Tollhaus, wo sie noch ist.�

    �Was ist denn aus dem Bates geworden?� fragte Kags.

    �Er wird hier sein, sobald es dunkel geworden ist, und treibt sich
    solange herum, wo er kann. Die aus 'n Kr�ppel sitzen alle, und die
    ganze Schenkstube ist voll von Polizei; ich hab's mit meinen eigenen
    Augen gesehen.�

    �Da wird noch manch einer mit hineinverwickelt werden�, bemerkte Toby,
    sich auf die Lippen bei�end.

    �'s ist Schwurgerichtssaison,� sagte Kags, �und wenn Bolter gegen Fagin
    aussagt, was er ohne Zweifel tun wird, so baumelt der Jude bei Gott
    nach sechs Tagen.�

    �Ihr h�ttet nur die Leute toben h�ren sollen�, fuhr Chitling fort.
    �H�tten die Schuker nicht wie Teufel gefochten, so w�r ihnen Fagin vom
    Volk entrissen. Er sah aus wie durch Kot und Blut gezogen, denn einmal
    war er schon niedergeschlagen und hing sich an die Schuker, als wenn
    sie seine teuersten Freunde gewesen w�ren. Sie mu�ten ihn in die Mitte
    nehmen, und der andr�ngende w�tende Haufen war wie 'ne Herde rei�ender,
    nach seinem Blute lechzender W�lfe und l�rmte wie besessen, und die
    Weiber schrien, da� sie ihm das Herz aus'm Leibe rei�en wollten.�

    Alle drei sa�en einige Minuten entsetzt und schweigend da, als
    pl�tzlich auf der Treppe ein Ger�usch ert�nte und unmittelbar darauf
    Sikes' Hund hereinsprang. Sie liefen an das Fenster; er mu�te durch
    irgendeine �ffnung hereingekommen sein; sein Herr war jedoch nicht zu
    sehen.

    �Was ist dies?� sagte Toby, nachdem sie vom Fenster zur�ckgetreten
    waren. �Ich will doch hoffen, da� er nicht hierher kommt?�

    �Wenn er das gewollt h�tte, w�rd' er mit dem Hunde gekommen sein, der
    gerade so aussieht, als wenn er weit hergelaufen w�re�, meinte Kags.

    �Aber woher kann er gekommen sein?� fuhr Toby fort. �Hm! er hat Fremde
    in den andern H�usern gefunden, und hier ist er schon �fter gewesen.
    Aber warum kommt er ohne ihn?�

    �Er� (keiner nannte den M�rder bei seinem Namen), �er ist sicher �bers
    Wasser,� sagte Kags, �und er hat den Hund zur�ckgelassen, der sonst
    nicht so ruhig daliegen w�rde.�

    Als es dunkel geworden war, verschlossen sie den Fensterladen und
    z�ndeten Licht an. Die schrecklichen Ereignisse der beiden letzten Tage
    hatten sie mit Furcht und Entsetzen erf�llt. Sie schreckten bei jedem
    Laute zusammen und fl�sterten nur von Zeit zu Zeit ein paar Worte, als
    wenn das Gespenst der Ermordeten im Hause umginge. Pl�tzlich wurde laut
    an die Haust�r geklopft. Crackit sah aus dem Fenster und erbla�te.
    Sie berieten, und das Ergebnis war, da� er eingelassen werden m�sse.
    Crackit ging und kehrte bald darauf mit einem Manne zur�ck, der mehr
    wie des M�rders f�rchterlicher Geist als wie Sikes selber aussah, mit
    seinen erdfahlen, eingefallenen Wangen, erloschenen, tiefliegenden
    Augen und langgewachsenem Barte. Er wollte sich auf einen Stuhl am
    Tische niederlassen, schauderte aber und schob den Stuhl dicht an die
    Wand.

    Kein Wort war noch gesprochen worden. Seine Blicke schweiften von
    dem einen zum andern. Ward ein Auge aufgeschlagen und begegnete dem
    seinigen, so wurde es augenblicklich wieder gesenkt. Als er endlich
    das Stillschweigen brach, schreckten alle drei bei dem nie vernommenen
    hohlen Tone seiner Stimme zusammen.

    �Wie kam der Hund hier ins Haus?� fragte er.

    �Allein. Vor drei Stunden.�

    �Es hei�t, da� Fagin eingezogen w�re. Ist's wahr oder gelogen?�

    �Vollkommen wahr.�

    Es trat ein abermaliges Schweigen ein.

    �Geht alle zur H�lle!� hub Sikes endlich, mit der Hand �ber die Stirn
    fahrend, wieder an. �Habt ihr mir nichts zu sagen?�

    Es erfolgte eine unruhige Bewegung unter ihnen, aber niemand sprach.

    �Ihr, der Ihr hier Herr im Hause spielt,� fuhr Sikes zu Crackit
    gewandt, fort, �denkt Ihr mich zu verkaufen oder mich hier unterducken
    zu lassen, bis die Hetze vorbei ist?�

    �Ihr k�nnt bleiben, wenn Ihr Euch hier f�r sicher haltet�, antwortete
    Toby z�gernd.

    Sikes blickte oder machte vielmehr nur den Versuch, hinter sich an der
    Wand hinaufzublicken, und sagte: �Ist -- ist sie -- ist die Leiche schon
    beigesetzt?�

    Das Kleeblatt sch�ttelte die K�pfe.

    �Warum nicht?� fuhr er, ebenso hinter sich blickend, fort. �Warum
    lassen sie ein so h��liches Ding �ber der Erde? -- Wer klopft da?�

    Toby erwiderte, es w�re nichts zu f�rchten, ging hinaus und trat mit
    Charley Bates wieder herein. Sikes sa� der T�r gegen�ber, so da� die
    Blicke des Knaben sogleich auf seine Gestalt fielen.

    �Toby,� sagte Charley, �warum habt Ihr mir das unten nicht gesagt?�

    Sikes sah die drei zusammenschrecken und hielt dem Knaben die Hand
    zutunlich schmeichelnd entgegen, denn es bem�chtigte sich seiner ein
    unnennbares Entsetzen.

    �La� mich in ein anderes Zimmer gehen�, sagte Charley, sich
    zur�ckziehend.

    �Charley�, sagte Sikes, aufstehend und ein paar Schritte vortretend:
    �wie -- kennst du mich nicht?�

    �Kommt mir nicht n�her!� rief der Knabe, noch weiter zur�ckweichend und
    schaudernd dem M�rder in das Angesicht blickend. �Ihr Ungeheuer -- Ihr
    Unmensch!�

    Sikes stand auf halbem Wege still, und beide blickten einander an; aber
    dann senkte der M�rder allm�hlich die Augen zu Boden.

    �Ich nehm' euch drei zu Zeugen,� rief der Knabe, die geballte Faust
    sch�ttelnd und im Fortreden einen immer heftigeren Ton annehmend,
    �ich nehm' euch drei zu Zeugen, da� ich mich nicht vor ihm f�rchte,
    und wird er hier gesucht, so zeig' ich ihn selbst an. Ich sag's euch
    rund heraus, er kann mich totschlagen, wenn's ihm beliebt, oder wenn
    er's wagt, aber bin ich hier, so zeig' ich ihn selbst an, w�rd' ihn
    anzeigen, und wenn er lebendig ger�stet werden sollte. Hilfe! M�rder!
    Wenn ihr nicht alle drei elende Memmen seid, so steht ihr mir bei.
    Hilfe! M�rder! Nieder mit ihm!�

    Er warf sich bei diesen Worten allein auf den riesenstarken Mann, und
    zwar so w�tend und pl�tzlich, da� beide zu Boden st�rzten. Die drei
    Zuschauer waren wie bet�ubt, machten nicht einmal Miene, sich in das
    Mittel zu legen, und der Knabe und der Mann w�lzten sich um und um,
    indem jener der auf ihn herabregnenden Streiche nicht achtete, die
    Kleider des M�rders immer fester vor der Brust desselben fa�te, und
    nicht aufh�rte, aus aller Macht nach Hilfe zu rufen.

    Der Kampf war jedoch zu ungleich, um lange w�hren zu k�nnen. Sikes
    hatte seinen Gegner unter sich gebracht und setzte ihm das Knie auf
    die Kehle, als ihn Crackit mit best�rzter Miene emporri� und nach dem
    Fenster hinwies. Es schimmerten Lichter auf der Stra�e unten, eifrige
    Stimmen ert�nten, von der n�chsten Br�cke her wurde der unaufh�rliche
    Schall von Fu�tritten vernommen, wie wenn eine zahllose Menschenmenge
    her�berk�me, unter welcher sich ein Berittener zu befinden schien,
    denn man h�rte deutlich das Ger�usch von Rossehufen auf dem unebenen
    Steinpflaster. Es wurde immer heller, der Nahenden Anzahl immer gr��er,
    und endlich wurde laut an die Haust�r geklopft, w�hrend ein heiseres
    Gemurmel unz�hliger zorniger Stimmen auch wohl den Beherztesten mit
    Beben erf�llte.

    �Hilfe, zu Hilfe!� schrie der Knabe im durchdringendsten Tone. �Hier
    ist er, hier ist er! Brecht die T�r auf!�

    �In des K�nigs Namen!� wurde drau�en gerufen und wiederum erhob sich,
    nur lauter, das zornige Gemurmel.

    �Schlagt die T�r ein!� schrie Charley. �Sie �ffnen sie nimmermehr.
    Schlagt die T�r ein und dann herauf, wo das Licht ist!�

    Ein lautes Hussa ert�nte, und es war, als wenn mit hundert und abermals
    hundert Kn�tteln und Stangen gegen die Fensterl�den geh�mmert w�rde.

    �Macht mir das Loch da auf, da� ich diesen verfluchten kleinen
    schreienden Galgenstrick einschlie�en kann�, rief Sikes w�tend,
    schleuderte den Knaben in ein Gemach hinein, das Toby �ffnete, und
    verschlo� es. �Ist die T�r unten gut verwahrt?�

    �Verschlossen und doppelt und dreifach verriegelt�, erwiderte Crackit,
    der gleich den andern beiden kaum einen deutlichen Gedanken fassen zu
    k�nnen schien.

    �Wie ist's mit den W�nden und Fenstern?� fragte Sikes weiter.

    �Verwahrt wie ein Gef�ngnis.�

    Jetzt �ffnete Sikes das Fenster und rief trotzig hinunter: �Seid alle
    verdammt! Macht eure Sachen, so gut ihr k�nnt, ihr bekommt mich doch
    nicht!�

    Ersch�tterndes Geschrei der w�tenden Menge erf�llte die Luft. Einige
    riefen, man m�ge das Haus anz�nden, andere, die Polizeidiener m�chten
    den M�rder totschie�en, und niemand zeigte eine solche unsinnige
    Wut wie der Reiter, der aus dem Sattel sprang, sich durch die Menge
    hindurchdr�ngte, als wenn er nur Wasser teilte, und dicht vor dem
    Hause mit einer den gr��lichen L�rm �bert�nenden Stimme rief: �Zwanzig
    Guineen, wer eine Leiter bringt.�

    Und nunmehr riefen Hunderte nach Leitern und Schmiedeh�mmern, andere
    rannten mit Fackeln hin und her, und noch andere stie�en Fl�che und
    Verw�nschungen aus, dr�ngten wie rasend gegen die T�r oder versuchten
    an dem Hause emporzuklimmen.

    �Es war Flutzeit, als ich kam�, rief Sikes, das Fenster wieder
    verschlie�end. �Gebt mir 'nen Strick. Sie sind alle vorn. Ich kann mich
    hinten in den Graben 'nunterlassen und entkommen. 'nen Strick -- hurtig
    -- oder ich tue noch drei Mordtaten und mache mir dann selber den
    Garaus.�

    Die drei von dem Schrecken Gel�hmten wiesen nach einem Winkel hin, in
    welchem Stricke lagen. Sikes w�hlte hastig den st�rksten und l�ngsten
    aus, eilte hinauf und bestieg das Dach.

    Der eingesperrte Knabe hatte unterdes nicht aufgeh�rt zu schreien
    und zu rufen, man m�chte das Haus von allen Seiten bewachen. Als der
    M�rder daher aus der Dachluke herausstieg, wurde er sogleich bemerkt,
    da Hunderte bereits Wege gesucht hatten, um nach dem Hinterhause zu
    gelangen. Die Ebbe war eingetreten, und er sah, da� der Graben nur mit
    Schlamm gef�llt war. Die Fenster und D�cher aller Hinterh�user umher
    waren bereits lebendig, und von oben und unten und allen Seiten ert�nte
    Triumphgeschrei, da� er nicht entrinnen k�nne.

    �Nun haben sie ihn, hurra!� schrie ein Mann auf der n�chsten, unter der
    Menschenwucht sich beugenden Br�cke, und ein tausendfaches Hurra hallte
    durch die Luft wider.

    �Ich gelobe demjenigen f�nfzig Pfund,� rief ein alter, gleichfalls
    auf der Br�cke stehender Herr, �der ihn lebendig greift. Ich will
    hierbleiben und zahle die Summe auf der Stelle.�

    Ein abermaliges allgemeines Geschrei ert�nte, in das sich der Ruf
    mischte, die T�r sei endlich erbrochen; der Menschenstrom flutete
    nun wieder zu dieser zur�ck, denn jeder wollte den M�rder von den
    Polizeibeamten herausbringen sehen. Es entstand das furchtbarste
    Gedr�nge, und das Dach wurde f�r den Augenblick weniger beachtet.

    Der M�rder, der, bereits verzweifelnd, unschl�ssig dagesessen hatte,
    fa�te jetzt wieder Hoffnung und beschlo�, den letzten Rettungsversuch
    zu wagen und sich auf die Gefahr, im Schlamme zu ersticken, in den
    Graben hinabzulassen, um wom�glich mit Hilfe der Dunkelheit und
    Verwirrung zu entfliehen. Die Hoffnung gab ihm neue Kraft, der sich
    ihm n�hernde L�rm im Hause stachelte ihn noch mehr an, er sprang auf,
    erreichte in zwei Augenblicken den Schornstein, befestigte das eine
    Ende seines Strickes an demselben und hatte im Nu an dem andern eine
    starke Laufschlinge gekn�pft. Er konnte sich mit dem Stricke fast bis
    auf Mannesl�nge hinunterlassen und nahm sein Messer zur Hand, um ihn
    zur rechten Zeit abzuschneiden und sich in den Graben zu werfen.

    In demselben Augenblicke, als er die Schlinge �ber den Kopf warf, um
    sie unter den Armen zu befestigen, und indem der erw�hnte alte Herr
    laut rief, der M�rder sei im Begriff, sich hinunterzulassen, blickte er
    hinter sich, schlug die H�nde �ber dem Kopfe zusammen und stie� einen
    lauten Schrei des Entsetzens aus. �Die Augen -- da sind sie wieder!�
    rief er mit hohler Grabesstimme, wankte, wie von einem Blitzstrahle
    getroffen, verlor das Gleichgewicht und taumelte vom Dache herunter,
    die Schlinge war an seinem Halse, und seine Schwere bewirkte, da� sie
    straff wie eine Bogenschnur und schnell wie ein Pfeil hinauflief. Er
    fiel f�nfunddrei�ig Fu� -- ein pl�tzlicher Ruck -- ein krampfhaftes
    Gliederzucken -- und da hing er mit dem offenen Messer in der
    zusammengepre�ten, steif werdenden Hand.

    Der alte Schornstein bebte von der Ersch�tterung, hielt sie jedoch
    aus. Der entseelte M�rder schwebte hin und wieder; Charley, dem er
    die Aussicht versperrte, stie� ihn zur Seite und rief, da� man seiner
    Gefangenschaft ein Ende machen m�chte; der Hund lief mit schrecklichem
    Geheul auf dem Dache hin und her und sprang endlich hinunter auf die
    Schulter des Erh�ngten, vermochte sich aber nicht festzuhalten, st�rzte
    und lag gleichfalls darauf tot da, denn er war mit dem Kopfe gegen
    einen spitzen Stein gefallen.




    51. Kapitel.

        Enth�llung mehr als eines Geheimnisses und ein Heiratsantrag ohne
        Erw�hnung eines Leibgedinges oder Nadelgeldes.


    Zwei Tage nach den im vorigen Kapitel erz�hlten Ereignissen befanden
    sich Mrs. Maylie und Rose, Oliver und der gute Doktor, Mr. Brownlow und
    Mrs. Bedwin und noch jemand auf der Reise nach Olivers Geburtsstadt.
    Oliver vermochte nur schwer seine Gedanken zu sammeln, und den �brigen
    erging es keineswegs besser. Brownlow hatte ihn und die beiden Damen
    mit den Monks abgepre�ten Aussagen genau bekannt gemacht; und obwohl
    sie wu�ten, da� der Zweck ihrer Reise in der Vollendung des so
    gl�cklich angefangenen Werkes bestand, so war doch die ganze Sache noch
    in ein so betr�chtliches Dunkel geh�llt, da� die gr��te Spannung sie
    folterte, obgleich Brownlow und Losberne die Schreckensauftritte der
    letzten Tage f�r jetzt noch verborgen vor ihnen gehalten hatten; und so
    setzten sie denn ihre Reise schweigend miteinander fort.

    Sie gelangten w�hrenddessen auf die Stra�e, auf welcher Oliver
    einst entflohen war, und mit Lebhaftigkeit erneuerte sich ihm die
    Erinnerung an jene Leidenszeit. �Sehen Sie, da, da!� rief er in der
    h�chsten Erregtheit, Roses Hand ergreifend und aus dem Wagenfenster
    hinauszeigend. �Das ist der Steg, �ber den ich hin�bersprang, und das
    ist die Hecke, hinter welcher ich fortschlich, und das ist der Fu�pfad,
    der nach dem Hause f�hrt, wo ich mich als kleines Kind aufhielt. O
    Dick, mein lieber Dick, wenn ich dich doch jetzt auch sehen k�nnte!�

    �Du wirst ihn bald sehen�, sagte Rose; �sollst ihm sagen, wie gl�cklich
    du geworden bist und wie dich nichts so sehr freute, als da� du
    gekommen seiest, um ihn an deinem Gl�cke teilnehmen zu lassen.�

    �Ja, ja! Und wir wollen ihn mit uns fortnehmen, ihn kleiden, und er mu�
    an einen guten Ort, wo er stark und gesund werden kann -- nicht wahr?�

    Rose nickte bejahend, denn der Knabe l�chelte durch so selige Tr�nen,
    da� sie keines Wortes m�chtig war.

    �Sie werden freundlich und liebevoll zu ihm sein,� fuhr Oliver fort,
    �denn Sie sind es gegen jedermann. Ich wei� es, Sie werden weinen
    m�ssen, wenn Sie h�ren, was er Ihnen wird erz�hlen k�nnen, aber auch
    wieder l�cheln, wie Sie es bei mir taten, als ich so ganz anders
    geworden war. Er sagte, als ich fortlief: >Geh mit Gott, Gottes Segen
    begleite dich!< oh, und ich will nun sagen: >Gottes Segen sei mit dir<,
    und ihm zeigen, wie lieb ich ihn daf�r habe!�

    Als sie endlich durch die engen Stra�en der Stadt fuhren, war Oliver
    wie au�er sich. Da war des Leichenbestatters Haus, nur weit kleiner und
    lange nicht so stattlich, wie es ihm vormals erschienen war, und alle
    wohlbekannten Geb�ude und L�den und Gamfields Karren wie sonst vor dem
    Gasthause, und das Armenhaus, das traurige Gef�ngnis seiner Kinderzeit,
    mit den d�steren Fenstern, und am Tore stand derselbe hagere Pf�rtner.
    Oliver schreckte unwillk�rlich zur�ck, lachte �ber sich selbst, so
    t�richt zu sein, und weinte und lachte aber- und abermals. So viele
    Gesichter an den T�ren und Fenstern erkannte er wieder; es war ihm,
    als wenn er die Stadt erst gestern verlassen h�tte und als wenn seine
    letzte Zeit nur ein gl�cklicher Traum gewesen w�re.

    Allein sie und die Gegenwart waren Wirklichkeit. Die Reisenden fuhren
    vor dem ersten Gasthause vor, das Oliver einst wie einen Palast
    angestaunt hatte, und Mr. Grimwig empfing sie dienstbeflissen, k��te
    die alte und junge Dame und war lauter L�cheln und Freundlichkeit, als
    wenn er aller Gro�vater w�re und nicht von fern daran d�chte, seinen
    Kopf aufzuessen, nicht einmal, als er einem sehr alten Postillon
    widersprach und es besser zu wissen behauptete, welcher Weg der n�chste
    nach London w�re, obwohl er denselben nur ein einziges Mal gekommen,
    und obendrein im Schlafe. Die Zimmer und das Mittagessen standen
    bereit, und alles war wie durch Zauber geordnet.

    Man kleidete sich um, kam wieder zusammen, und dieselbe Stille und
    Zur�ckhaltung begann wieder zu herrschen. Brownlow erschien nicht beim
    Mittagessen; die beiden andern Herren liefen mit wichtigen Mienen aus
    und ein und fl�sterten miteinander, wenn sie im Zimmer waren. Endlich
    wurde auch Mrs. Maylie hinausgerufen und kehrte erst nach einer Stunde
    mit rotgeweinten Augen zur�ck. Dieses alles versetzte Rose und Oliver,
    die in die neuesten Geheimnisse nicht eingeweiht waren, in gro�e
    Unbehaglichkeit und Unruhe. Sie sa�en stumm da, und wenn sie bisweilen
    ein paar Worte sprachen, so geschah es fl�sternd, als ob sie den Laut
    ihrer eigenen Stimmen f�rchteten.

    Es war neun Uhr geworden, und sie fingen an zu glauben, da� sie vor
    morgen nichts mehr h�ren w�rden, als die Herren Losberne, Grimwig und
    Brownlow mit einem Manne hereintraten, bei dessen Erblicken Oliver im
    Begriff war laut aufzuschreien. Sie sagten ihm, es w�re sein Bruder,
    und es war derselbe, den er in dem St�dtchen mit Fagin am Fenster
    seines kleinen Zimmers gesehen hatte. Monks oder Charles Leeford setzte
    sich unweit der T�r und konnte sich auch jetzt nicht enthalten, dem
    erstaunten Knaben einen Blick giftigen Grolls zuzuwerfen. Brownlow
    trat mit Papieren in der Hand an den Tisch, in dessen N�he Rose und
    Oliver sa�en.

    �Diese in London vor mehreren Herren unterzeichneten Erkl�rungen�, hub
    er an, �m�ssen im wesentlichen hier wiederholt werden, so peinlich
    es allen Beteiligten auch sein mag. Ich h�tte Ihnen die Dem�tigung
    gern erspart, allein es ist notwendig, da� wir Ihre Aussage aus Ihrem
    eigenen Munde h�ren, und Sie wissen, warum.�

    �Fahren Sie fort�, sagte Monks, sich abwendend. �Rasch. Ich habe genug
    getan. Halten Sie mich nicht auf!�

    �Dieser Knabe�, sprach Brownlow weiter, die Hand auf Olivers Kopf
    legend, �ist Ihr Halbbruder, der Sohn Ihres Vaters, meines teuren
    Freundes Edwin Leeford, von der jungen Agnes Fleming, der die Geburt
    des Kindes das Leben kostete.�

    �Ja�, sagte Monks, dem zitternden Knaben, dessen Herzschl�ge er fast zu
    h�ren meinte, fortw�hrend finster-grollende Blicke zuwerfend. �Er ist
    der Bastard.�

    �Der Ausdruck, dessen Sie sich bedienen�, entgegnete Brownlow im Tone
    strengen Tadels, �enth�lt einen Vorwurf gegen Verstorbene, die den
    kurzsichtigen Richterspr�chen dieser Welt l�ngst entr�ckt sind, und
    beschimpft keinen Lebenden, Sie selbst ausgenommen. Doch genug davon.
    Der Knabe ist in dieser Stadt geboren?�

    �Im Armenhause dieser Stadt. Sie haben es da aufgezeichnet.�

    �Die hier Anwesenden m�ssen es auch h�ren.�

    �So h�ren Sie. Als sein Vater in Rom erkrankt war, begab sich seine
    Frau, meine Mutter, zu ihm -- soviel ich wei�, um sein Verm�gen an
    sich zu nehmen, denn sie hatte keine Zuneigung zu ihm, wie er nicht
    zu ihr. Sie nahm mich mit. Er kannte uns nicht. Denn er lag schon
    ohne Bewu�tsein und schlummerte bis zum folgenden Tage fort, an dem
    er starb. In seinem Schreibtische befand sich ein P�ckchen Papiere,
    datiert vom ersten Tage seiner Krankheit und adressiert an Sie, Mr.
    Brownlow, mit der Bemerkung, da� es erst nach seinem Tode zu bef�rdern
    sei. Das P�ckchen enthielt ein Schreiben an Agnes Fleming und ein
    Testament. Das Schreiben war voll von reuigen Bekenntnissen seiner
    gegen sie angewandten Verf�hrungsk�nste und Bitten zu Gott um Beistand
    f�r sie. Es fehlten zu der Zeit nur noch ein paar Monate bis zu ihrer
    Entbindung. Er sagte ihr, was er zu tun beabsichtigte, ihre Unehre
    zu verbergen, wenn er am Leben bliebe, und flehte sie an, falls er
    sterbe, seinem Andenken nicht zu fluchen, oder zu glauben, da� sein
    und ihr Vergehen an ihr und ihrem Kinde heimgesucht werden w�rde, denn
    die ganze Schuld w�re sein. Er erinnerte sie an den Tag, an welchem er
    ihr das kleine Schlo� geschenkt und den Ring mit ihrem Taufnamen und
    einem offenen Raume f�r den Namen, den er gehofft auf sie �bertragen
    zu k�nnen; bat sie, das Geschmeide, wie sonst, auf ihrem Herzen zu
    bewahren, und wiederholte dann das alles aber- und abermals, als wenn
    er von Sinnen gewesen w�re -- was, wie ich glaube, auch wirklich der
    Fall gewesen ist.�

    �Aber das Testament�, fiel Brownlow ein, da Oliver schmerzliche Z�hren
    �ber die Wangen hinabliefen, �war in demselben Sinne und Geiste
    abgefa�t. Er sprach darin von dem Elende, das ihm seine Frau bereitet,
    von der fr�hen Bosheit und Ruchlosigkeit seines einzigen in Ha� gegen
    ihn erzogenen Sohnes und vermachte Ihnen und Ihrer Mutter Jahrgelder
    von je achthundert Pfund. Die Masse seines Verm�gens teilte er in zwei
    gleiche Teile und bestimmte den einen f�r -- Agnes Fleming, und den
    andern f�r sein und ihr Kind, wenn es lebendig geboren w�rde und die
    Jahre der M�ndigkeit erreichte. Wenn es ein M�dchen w�re, so sollte ihm
    die Erbschaft bedingungslos zufallen; w�re es aber ein Knabe, so sollte
    sie an die Bedingung gekn�pft sein, da� der Erbe bis zu den Jahren
    der M�ndigkeit seinen Namen durch keinerlei �ffentliches Vergehen
    befleckte. Ihr Vater traf diese Bestimmung, wie er sagte, um dadurch
    sein Vertrauen zu der Mutter und seine, durch den herannahenden Tod nur
    verst�rkte �berzeugung darzulegen, da� ihr Kind ihre Tugenden, ihre
    edlen Gesinnungen erben w�rde. Wenn seine Voraussetzung nicht eintr�fe,
    sollte das Geld Ihnen zufallen; denn nur, wenn beide Kinder einander
    gleich w�ren, sollte Ihr fr�herer Anspruch auf sein Verm�gen anerkannt
    sein, der Sie keine Anspr�che auf sein Herz, und ihm von der fr�hesten
    Kindheit an K�lte und Abneigung bewiesen h�tten.�

    �Meine Mutter�, nahm Monks, und zwar mit lauterer Stimme, wieder das
    Wort, �tat, was einer Mutter zukam -- sie verbrannte dieses Testament.
    -- Das Schreiben gelangte nie an seine Adresse, sie bewahrte es aber
    nebst anderen Dokumenten auf, falls die Flemings den Versuch machen
    sollten, den Makel abzuleugnen. Agnes' Vater vernahm die Wahrheit �ber
    sie mit jeder �bertreibung und Vergr��erung, die ihr bitterer Ha� --
    wof�r ich sie jetzt liebe, hinzuzuf�gen vermochte. Sein verletztes
    Ehrgef�hl bewog ihn, sich mit seinen Kindern nach einem entlegenen
    Orte in Wales zu begeben und, um desto gewisser selbst seinen Freunden
    verborgen zu bleiben, sogar seinen Namen zu ver�ndern. Er wurde nicht
    lange darauf tot in seinem Bette gefunden. Die Tochter war einige
    Wochen zuvor heimlich entwichen. Er hatte selbst die Umgegend nach ihr
    durchstreift, war aber mit der �berzeugung zur�ckgekehrt, da� sie sich
    den Tod gegeben, und �berlebte seinen Kummer nur wenige Stunden.�

    Es trat ein kurzes Stillschweigen ein, bis Brownlow den Faden der
    Erz�hlung wieder aufnahm. �Nach Jahren,� sagte er, �erschien dieses
    jungen Mannes -- Eduard Leefords -- Mutter bei mir. Er hatte sie in
    seinem achtzehnten Jahre verlassen, sie ihrer Juwelen und ihres Geldes
    beraubt, hatte gespielt, vergeudet, gef�lscht und war nach London
    gegangen, wo er sich dem schlechtesten Gesindel zugesellte. Sie litt
    an einer schmerzhaften und unheilbaren Krankheit und w�nschte ihn vor
    ihrem Tode noch wiederzusehen. Es wurden die genauesten Nachforschungen
    angestellt, welche endlich Erfolg hatten. Er ging mit ihr nach
    Frankreich zur�ck.�

    �Und sie starb dort,� fiel Monks ein, �nachdem sie lange auf dem
    Siechbette gelegen; kurz vor ihrem Tode vermachte sie mir diese
    Geheimnisse, samt ihrem unausl�schlichen und t�dlichen Hasse gegen
    alle in diese Angelegenheit Verwickelten, was jedoch unn�tig war, denn
    er lebte schon seit langer Zeit in mir. Sie wollte es nicht glauben,
    da� das M�dchen sich und dem Kinde den Tod gegeben, sondern hielt
    sich �berzeugt, da� ein Knabe geboren und am Leben w�re. Ich schwur
    ihr, wenn je das Dasein eines solchen zu meiner Kunde gelangte, ihm
    nachzusp�ren, ihm nimmer Ruhe zu lassen, ihn mit der bittersten,
    unvers�hnlichsten Feindschaft zu verfolgen, allen Ha� an ihm
    auszulassen, dessen mein Innerstes f�hig war -- ihn, den hochtrabenden
    Worten des beleidigenden Testaments zum Hohne, an den Galgen zu
    bringen. Sie hatte recht gehabt. Er kam mir endlich in den Weg; ich
    machte einen guten Anfang -- und w�rde -- ja, w�rde geendet haben,
    wie begonnen, wenn nicht eine schwatzm�ulige Trulle meine Anschl�ge
    vereitelt h�tte!�

    Der Sch�ndliche schlug sich mit der Hand vor die Stirn, murmelte in
    der Wut ohnm�chtiger Bosheit Verw�nschungen �ber sich selbst; Brownlow
    wandte sich unterdessen zu seinen entsetzten Freunden und sagte ihnen,
    da� der Jude, der Leefords alter Vertrauter und Helfershelfer gewesen
    w�re, eine gro�e Belohnung von ihm f�r Olivers Umstrickung erhalten
    h�tte. Es sei ausbedungen gewesen, da� er einen Teil der gezahlten
    Summe zur�ckerstatten solle, falls Oliver wieder frei w�rde, und ein
    Streit �ber diesen Punkt habe beide auf das Land gef�hrt, welche Reise
    den Zweck gehabt, zu erkunden, ob der von Mrs. Maylie aufgenommene
    Knabe wirklich Oliver sei.

    �Was haben Sie von dem Schlosse und Ringe zu sagen?� fragte Brownlow,
    Monks wieder anredend.

    �Ich kaufte sie den Leuten ab, von welchen ich Ihnen gesagt habe. Sie
    hatten sie der W�rterin entwandt, die sie der Leiche abgenommen hatte�,
    erwiderte Monks, ohne die Augen aufzuschlagen. �Sie wissen, was daraus
    geworden ist.�

    Brownlow gab Grimwig einen Wink, dieser eilte voller Eifer hinaus und
    kehrte nach wenigen Augenblicken mit dem widerstrebenden Ehepaare aus
    dem Armenhause zur�ck.

    �Tr�gen mich meine Augen, oder sehe ich den kleinen Oliver wirklich
    vor mir?� rief Mr. Bumble mit schlecht erk�nsteltem Entz�cken. �Ach,
    Oliver, wenn du w��test, wie bek�mmert ich um dich gewesen bin!�

    �Schweig, Dummkopf!� murmelte seine Eheh�lfte.

    �Frau, kann ich meinen Gef�hlen wehren,� entgegnete er, �wenn ich, der
    ich ihn kirchspielm��ig erzogen habe, ihn sitzen sehe zwischen den
    allerangenehmsten Damen und Herren? Ich hatte den Knaben immer so lieb,
    als wenn er mein eigner -- eigner Gro�vater gewesen w�re�, sprudelte
    Mr. Bumble heraus, nachdem er m�hsam dem passendsten Vergleich
    nachgesonnen hatte. �Master Oliver, mein guter Oliver, erinnerst du
    dich noch des lieben, braven Herrn mit der wei�en Weste? Ach, er schied
    vorige Woche von der Erde in den Himmel, mit einem eichenen Sarge mit
    plattierten Griffen, Oliver.�

    �Seien Sie so gut, Ihre Gef�hle f�r sich zu behalten, Sir�, fiel
    Grimwig bissig ein.

    �Ich will mein m�glichstes tun, Sir�, erwiderte Bumble und wandte sich
    zu Brownlow: �Wie befinden Sie sich, Sir? Hoffentlich sehr wohl.�

    Brownlow beachtete seine Frage nicht, trat dicht vor das Ehepaar, wies
    nach Monks und fragte seinerseits: �Kennen Sie den Mann?�

    �Nein�, antwortete Frau Bumble keck.

    �Vielleicht kennen Sie ihn, Mr. Bumble?�

    �Ich habe ihn nie in meinem Leben gesehen.�

    �Ihm auch nichts verkauft?�

    �Nein�, sagte Frau Bumble.

    �Hatten Sie nicht einmal ein goldenes Schlo� und einen Ring?�

    �Beh�te. Sind wir denn blo� hier, um so l�ppische Fragen zu
    beantworten?�

    Brownlow gab Grimwig abermals einen Wink, und abermals enteilte Grimwig
    mit ungemeinem Eifer und kehrte mit zwei alten, wankenden, gichtischen
    Frauen zur�ck.

    �Sie verschlossen die T�r an dem Abend, da die alte Sally starb,
    konnten aber die Ritzen nicht verstopfen�, sagte die eine, ihre welke
    Hand emporhebend, und die andere stimmte bei.

    �Wir h�rten,� fuhr die erste fort, �da� sie Ihnen sagen wollte, was
    sie getan hatte, und sahen, da� Sie ihr etwas in Papier aus der Hand
    nahmen, und am anderen Tage, da� Sie zum Pfandleiher gingen.�

    �Ja,� f�gte die zweite hinzu, �und wir sp�rten auch aus, da� in dem
    Papiere ein goldenes Schlo� und ein Ring gewesen war.�

    �Und wissen noch mehr�, sprach die erste weiter. �Die alte Sally hat
    uns oft erz�hlt, die junge Frauensperson h�tte ihr gesagt, da� sie
    gef�hlt h�tte, sie w�rde es nicht �berleben, und w�re zur Zeit, da sie
    den Knaben geboren, auf dem Wege gewesen, um am Grabe des Vaters ihres
    Kindes zu sterben.�

    �Wollen Sie den Pfandleiher sehen?� fragte Grimwig, mit der Hand auf
    dem T�rgriffe.

    �Nein�, antwortete Frau Bumble. �Da er� -- sie wies nach Monks -- �so
    memmenhaft gewesen ist, zu bekennen, wie ich sehe, da� er es gewesen,
    und da Sie aus all den alten Hexen die rechten herausgesp�rt, so habe
    ich nichts mehr zu sagen. Ja, ich verkaufte die alten Scharteken; sie
    liegen, wo Sie sie nimmer wiederfinden werden, und was nun mehr?�

    �Oh, nichts weiter,� sagte Brownlow, �als da� es jetzt unsere Sache
    ist, Sorge zu tragen, da� man Ihnen und Ihrem Manne kein Vertrauen als
    Beamten mehr schenkt. Sie k�nnen gehen.�

    �Ich will doch hoffen,� nahm Bumble best�rzt das Wort, als Grimwig die
    beiden alten Frauen hinausf�hrte, �da� mich dieser ungl�ckliche kleine
    Umstand meines Kirchspieldienstes nicht berauben wird?�

    �Das wird er allerdings,� erwiderte Brownlow, �und Sie k�nnen sehr froh
    sein, wenn Sie noch so davonkommen.�

    Frau Bumble entfernte sich, und sobald sie die T�r hinter sich
    geschlossen hatte, erkl�rte ihr Eheherr, da� sie alles getan und sich
    davon nicht h�tte zur�ckhalten lassen wollen.

    �Das ist keine Entschuldigung�, sagte Brownlow. �Sie waren gegenw�rtig
    bei dem Verkaufe und sind vor dem Gesetze der noch schuldigere Teil, da
    Ihre Frau gem�� demselben unter Ihrer Leitung handelt.�

    �Wenn das Gesetz so lautet,� sagte Bumble, seinen Hut pathetisch
    zusammendr�ckend, �so ist es ein Esel -- ein Einfaltspinsel. Wenn es so
    kurzsichtig ist, so ist's ein blo�er Junggesell, und ich w�nsche ihm
    das Aller�rgste -- n�mlich, da� ihm die Augen durch Erfahrung ge�ffnet
    werden m�gen -- ja, durch Erfahrung!�

    Er folgte nach diesen Worten seiner Eheh�lfte mit verzweifelter
    Resignation, und Brownlow wandte sich zu Rose.

    �Mein liebes Fr�ulein, reichen Sie mir Ihre Hand. Zittern Sie nicht;
    Sie k�nnen die wenigen Worte, welche wir noch zu sagen haben, ohne
    Furcht h�ren.�

    �Ich wei� nicht, ob sie Bezug auf mich haben k�nnen,� sagte Rose, �aber
    wenn -- wenn es der Fall ist, so lassen Sie sie mich ein andermal
    h�ren. Ich habe jetzt nicht die Kraft dazu.�

    �Sie sind st�rker, als Sie glauben�, wandte Brownlow ein; �ich wei� es.
    Kennen Sie diese junge Dame, Sir?�

    Monks bejahte.

    �Ich sah Sie nie�, sagte Rose mit bebender Stimme.

    �Ich habe Sie oft gesehen�, versetzte Monks.

    �Der ungl�cklichen Agnes' Vater hatte zwei T�chter�, fiel Brownlow ein.
    �Was war das Schicksal der anderen -- der j�ngsten?�

    �Als ihr Vater starb,� antwortete Monks, �an einem fremden Orte,
    unter angenommenem Namen, ohne das mindeste zu hinterlassen, was zur
    Auffindung ihrer Verwandten h�tte f�hren k�nnen, nahmen arme Leute
    sie zu sich und erzogen sie. Der Ha� sp�rt nicht selten auf, was der
    Liebe verborgen bleibt. Meine Mutter fand das Kind nach Jahresfrist.
    Die Leute waren arm und fingen an, ihrer aufopfernden Gro�mut m�de zu
    werden. Zum wenigsten war es bei dem Manne der Fall. Meine Mutter lie�
    ihnen das M�dchen daher, gab ihnen ein unbedeutendes Geschenk an Geld
    und versprach mehr, was sie aber nie zu schicken gedachte. Die Armut
    und Unzufriedenheit der Leute verhie�en, da� das Kind ungl�cklich genug
    werden w�rde, schienen meiner Mutter indes noch nicht ganz zu gen�gen.
    Sie erz�hlte ihnen daher die Geschichte der Schwester mit angemessenen
    Ver�nderungen und sagte ihnen, sie m�chten auf das Kind sorgf�ltig
    achten, denn es stamme von b�sem Blute her, w�re unehelich geboren und
    w�rde fr�her oder sp�ter auf �ble Wege geraten. Die Umst�nde schienen
    das alles zu best�tigen, die Leute glaubten es und behandelten das Kind
    so hart, wie wir es nur w�nschen konnten, bis der Zufall wollte, da�
    eine damals in Chester wohnende verwitwete Dame aus Mitleid es mit sich
    fortnahm. Es war, als wenn ein H�llenspuk uns genarrt h�tte, denn trotz
    all unserer Anstrengungen blieb des alten Fleming Tochter bei der Dame
    und war gl�cklich; ich verlor sie vor ein paar Jahren aus den Augen und
    sah sie erst vor wenigen Monaten wieder.�

    �Sehen Sie die junge Dame jetzt?�

    �Ja -- sie lehnt an Ihrem Arme.�

    Rose war einer Ohnmacht nahe. Mrs. Maylie schlo� sie in die Arme und
    rief aus: �Du bist und bleibst meine liebe Nichte -- mein �ber alles
    teures Kind. Ich m�chte dich um alle Sch�tze der Welt nicht verlieren.�

    �Sie sind die einzige Freundin, die ich jemals hatte,� schluchzte Rose,
    �sind mir stets die liebreichste, beste Mutter gewesen. O wie soll ich
    dieses alles ertragen!�

    �Du hast mehr erduldet und hast dich unter jeglichem Leid als das
    beste, herrlichste M�dchen gezeigt und von jeher alle froh und
    gl�cklich gemacht, die dich kannten. Aber schau hier, wer es ist, der
    sich sehnt, dich in die Arme zu schlie�en.�

    �Oh, ich werde sie niemals Tante nennen�, rief Oliver. �Meine
    Schwester, meine liebe Schwester. Es war etwas in meinem Herzen, das
    mich von Anfang an trieb, sie so innig zu lieben. O Rose, meine beste
    Rose!�

    M�gen die Tr�nen, welche geweint, die abgebrochenen Worte, die in der
    Umarmung der beiden Waisen gewechselt wurden, geheiligt sein! Ein
    Vater, eine Schwester und Mutter waren in demselben Augenblick gewonnen
    und verloren; Freude und Schmerz gemischt in der Schale; doch war keine
    Z�hre eine bittere, denn auch der Schmerz war so gemildert, und so
    s��e, wonnige Gedanken gesellten sich ihm, da� er in eine hohe Freude
    verwandelt wurde und ganz seinen Stachel verlor.

    Sie waren eine lange, lange Zeit allein. Endlich wurde leise geklopft,
    Oliver �ffnete die T�r, schlich hinaus und Harry Maylie stand im Zimmer.

    �Ich wei� alles�, sagte er, neben der lieblichen Jungfrau Platz
    nehmend. �Teure Rose, ich wei� alles, -- wu�te es gestern schon --
    und komme, dich an ein Versprechen zu erinnern. Du gabst mir die
    Erlaubnis, jederzeit innerhalb eines Jahres auf den Gegenstand unserer
    letzten Unterredung zur�ckzukommen -- nicht in dich zu dringen, deinen
    Entschlu� zu �ndern, dich ihn wiederholen zu h�ren, wenn du wolltest.
    Ich sollte dir zu F��en legen d�rfen, was ich bes��e, nur ohne den
    Versuch zu machen, wenn du bei deinem Beschlusse beharrtest, ihm untreu
    zu werden.�

    �Dieselben Gr�nde, welche mich damals bestimmten, bestimmen mich noch
    jetzt�, erwiderte Rose mit Festigkeit. �In welchem Augenblicke k�nnte
    ich lebhafter empfinden, was ich der edlen Frau schuldig bin, die mich
    von einem leiden- und vielleicht schmachvollen Leben errettet hat? Ich
    habe einen Kampf zu k�mpfen, bin aber stolz darauf, ihn zu bestehen; er
    ist ein schmerzlicher, aber mein Herz wird nicht erliegen.�

    �Die Enth�llungen dieses Abends --�

    �Lassen mich in bezug auf dich in derselben Lage.�

    �Du verh�rtest dein Herz gegen mich, Rose.�

    �O Harry, Harry,� sagte Rose, in Tr�nen ausbrechend, �ich wollte, da�
    ich es k�nnte, um mir diese Pein zu ersparen.�

    �Warum aber f�gst du sie dir selber zu?� entgegnete Harry, ihre Hand
    ergreifend. �Denk doch an das, was du heute abend vernommen, Rose!�

    �Ach, was habe ich vernommen! Da� mein Vater den ihm zugef�gten Schimpf
    tief genug empfand, um sich in g�nzliche Verborgenheit zur�ckzuziehen
    -- o Harry, wir haben genug gesagt.�

    �Noch nicht, noch nicht�, rief er, die Aufstehende zur�ckhaltend.
    �Meine Hoffnungen, W�nsche, Entw�rfe, Gef�hle -- alles in mir ist
    anders geworden, nur meine Liebe nicht. Ich biete dir jetzt keine
    Auszeichnung, keine gl�nzende Stellung mehr in einer verkehrten,
    trugvollen Welt, in welcher alles beschimpft, nur das wahrhaft
    Schandbare nicht; nein, nur einen stillen, bescheidenen h�uslichen
    Herd, liebste Rose, mehr habe ich nicht zu bieten.�

    �Was willst du damit sagen?� stammelte die junge Dame.

    �Als ich das letztemal von dir schied, verlie� ich dich mit dem
    festen Entschlusse, alle eingebildeten Schranken zwischen dir und mir
    niederzurei�en -- deine Welt zur meinigen zu machen, wenn die meinige
    nicht die deine sein k�nnte -- und dem Geburtsstolze den R�cken zu
    wenden, damit er nicht hochm�tig auf dich herabzuschauen verm�chte. Ich
    habe es getan. Die mich an sich zogen, entfernten sich von mir -- die
    mich anl�chelten, zeigen mir frostige Mienen. Wohl! es gibt lachende
    Gefilde und schattige B�ume in Englands sch�nster Grafschaft, dort
    neben einer Dorfkirche -- der meinigen, Rose, steht ein l�ndliches
    Haus, auf das du mich stolzer machen kannst, als es alle die Hoffnungen
    und Aussichten verm�gen, denen ich entsagt habe, entsagt haben w�rde,
    und wenn sie noch tausendmal lockender gewesen w�ren. Das ist jetzt
    mein Besitztum und mein Stand, meine Stellung in der Welt -- und ich
    lege alles vor dir nieder.�

           *       *       *       *       *

    �'s ist 'ne Geduldsprobe, mit dem Abendessen auf Verliebte zu warten�,
    sagte Grimwig, aus einem Schl�fchen erwachend.

    Die Wahrheit zu sagen, das Abendessen lie� ungeb�hrlich lange auf
    sich warten, und weder Mrs. Maylie noch Harry oder Rose (die zugleich
    erschienen) wu�ten auch nur ein Wort zur Entschuldigung zu sagen.

    �Ich dachte ernstlich daran, heute abend meinen Kopf aufzuessen,� sagte
    Grimwig, �denn ich fing an zu glauben, da� ich weiter nichts bekommen
    w�rde. Wenn Sie erlauben, so nehme ich mir die Freiheit, die angehende
    Braut zu k�ssen.�

    Er verlor keine Zeit, seine Ank�ndigung bei dem err�tenden M�dchen zur
    Ausf�hrung zu bringen, und sein Beispiel ermunterte den Doktor und
    Brownlow zur Nachfolge. Einige wollen wissen, Harry Maylie h�tte es
    selbst in einem ansto�enden dunkeln Zimmer gegeben, was jedoch von den
    besten Autorit�ten f�r arge Verleumdung erkl�rt wird, da er jung und
    ein Geistlicher gewesen w�re.

    �Mein lieber Oliver, wo bist du gewesen, und warum siehst du so traurig
    aus?� fragte Mrs. Maylie. �Wie -- Tr�nen in diesem Augenblicke?�

    Wir leben in einer Welt der T�uschungen. Wie oft sehen wir unsere
    liebsten -- die am meisten uns ehrenden Hoffnungen vereitelt!

    Der arme kleine Dick war tot.




    52. Kapitel.

        Des Juden letzte Nacht.


    Der Gerichtssaal war zum Ersticken gef�llt -- kein Auge, das nicht
    auf den Juden geheftet gewesen w�re. Der Vorsitzende erteilte den
    Geschworenen die Rechtsbelehrung. Mit gr��ter Spannung horchend stand
    Fagin da, die Hand am Ohre, um kein Wort zu verlieren. Bisweilen
    blickte er scharf nach den Geschworenen hin�ber, die Wirkung auch
    nur des im kleinsten ihm g�nstigen Worts zu erlauschen -- bisweilen
    angstvoll nach seinem Anwalt, wenn die Rede in ersch�tternder,
    furchtbarer Klarheit wider ihn zeugte. Sonst aber regte er weder Hand
    noch Fu� und verharrte noch in der Stellung des angstvoll Horchenden,
    nachdem der Richter seine Darlegung l�ngst beendigt.

    Ein leises Gemurmel rief ihn zum Bewu�tsein zur�ck. Er hob die Augen
    empor und sah die Geschworenen miteinander beraten. Alle Blicke waren
    auf ihn gerichtet, und man fl�sterte schaudernd miteinander. Einige
    wenige schienen ihn nicht zu beachten. In ihren Mienen dr�ckte sich
    unruhige Verwunderung aus, wie die Jury z�gern k�nne, ihr Schuldig
    auszusprechen; allein in keinem Antlitze -- sogar in keinem der
    zahlreich anwesenden Frauen -- vermochte er auch nur das leiseste
    Anzeichen des Mitleids zu lesen. Alle schienen mit Begier seine
    Verurteilung zu fordern.

    Abermals trat eine Totenstille ein -- die Geschworenen hatten sich an
    den Vorsitzenden gewandt. Horch!

    Sie baten nur um die Erlaubnis, sich zur�ckziehen zu d�rfen.

    Er forschte, als sie einer hinter dem andern hinausgingen, in ihren
    Mienen, wohin wohl die Mehrzahl neigen m�chte; allein vergeblich. Der
    Kerkermeister ber�hrte ihn an der Schulter. Er folgte ihm mechanisch in
    den Hintergrund der Anklagebank und lie� sich auf einen Stuhl nieder,
    den jener ihm wies, denn er w�rde ihn sonst nicht gesehen haben.

    Er schaute abermals nach den Zuh�rern. Einige a�en und andere wehten
    sich mit den T�chern K�hlung zu. Ein junger Mann zeichnete sein Gesicht
    in eine Brieftasche. Fagin dachte, ob die Zeichnung wohl �hnlich werden
    m�chte, und sah zu, als der Zeichner seinen Bleistift spitzte, wie es
    jeder unbeteiligte Zuschauer h�tte tun k�nnen.

    Er wandte sich nach dem Richter und begann sich innerlich mit dem
    Anzuge desselben zu besch�ftigen -- von welchem Schnitte er w�re und
    was er kosten d�rfe. Auf der Richterbank hatte ein alter, beleibter
    Herr gesessen, der sich entfernt hatte und jetzt zur�ckkehrte, und
    Fagin �berlegte, ob der Herr zu Mittag gespeist und wo er gesessen,
    und was dergleichen Gedanken mehr waren, bis ein neuer Gegenstand neue
    Gedanken in ihm erweckte.

    Trotzdem war freilich sein Gem�t keinen Augenblick von dem peinigenden
    und dr�ckenden Gef�hle frei, da� sich das Grab zu seinen F��en �ffnete;
    es schwebte ihm fortw�hrend vor, aber undeutlich und unbestimmt, und er
    vermochte seine Gedanken nicht dabei festzuhalten. Und so geschah es,
    da� er, w�hrend bald Fieberhitze ihn ergriff und es ihn bald mit kaltem
    Schauder �berlief, die eisernen St�be der Anklagebank z�hlte, die er
    vor sich sah, und bei sich selber dachte, wie es wohl gekommen sein
    m�chte, da� einer derselben abgebrochen w�re; und ob man wohl einen
    neuen einschlagen w�rde oder nicht. Dann schweiften seine Gedanken
    wieder zu den Schrecken des Galgens und Schafotts ab, bis ein Aufw�rter
    den Boden mit Wasser besprengte, was jenen abermals eine andere
    Richtung gab.

    Endlich wurde Stille geboten, und alle Blicke waren pl�tzlich auf die
    T�r gerichtet. Die Geschworenen kehrten zur�ck und gingen dicht an
    ihm vor�ber. Die Gesichter der Geschworenen waren wie von Stein, er
    vermochte nichts darin zu lesen. Es trat eine Stille ein -- vollkommen
    -- atemlos -- Schuldig!

    Der Saal hallte von einem ersch�tternden Rufe wider, der sich mehrmals
    wiederholte und durch ein donnerndes Geschrei beantwortet wurde, durch
    welches die Menge drau�en ihren Jubel ausdr�ckte, da� der Verurteilte
    am Montag sterben m�sse.

    Er wurde gefragt, ob er etwas zu sagen wisse, weshalb die
    Urteilsvollziehung nicht statthaben m�chte. Er hatte seine horchende
    Stellung wieder angenommen und blickte den Richter scharf an, der
    jedoch die Frage zweimal wiederholen mu�te, ehe der Jude sie zu
    vernehmen schien, der endlich nur murmelte, er w�re ein alter Mann --
    ein alter Mann -- ein alter Mann. Seine Stimme verlor sich in leises
    Fl�stern, und bald schwieg er g�nzlich.

    Der Richter setzte die schwarze M�tze auf, -- der Verurteilte stand
    noch immer da mit derselben Miene in derselben Stellung. Die ernste
    Feierlichkeit des Augenblicks pre�te einer Frau einen Ausruf aus --
    er blickte hastig und lauschend empor -- stand aber da gleich einer
    Bilds�ule, obgleich der Ton, das Wort, alle Anwesenden durchbebte. Er
    blickte noch immer starr vor sich hin, als ihm der Kerkermeister die
    Hand auf den Arm legte und ihm winkte. Er sah ihn einen Augenblick wie
    bet�ubt an und gehorchte.

    Er wurde hinunter in einen gepflasterten Raum gef�hrt, wo einige
    Angeklagte warteten, bis die Reihe an sie k�me, und andere sich mit
    ihren Freunden unterredeten, die sich an dem in den Hof �ffnenden
    vergitterten Fenster gesammelt hatten und unter denen niemand war,
    der mit ihm gesprochen h�tte, die aber alle bei seiner Ann�herung
    zur�cktraten, um ihn der Volksmenge drau�en hinter den Eisenst�ben
    sichtbarer zu machen; und er wurde laut mit Schimpfnamen, Geschrei und
    Gezisch begr��t. Er sch�ttelte die Faust und w�rde die N�chststehenden
    angespien haben, allein seine F�hrer dr�ngten ihn rasch fort durch
    einen d�steren, nur von wenigen matt brennenden Lampen erleuchteten
    Gang in das Innere des Gef�ngnisses, wo er durchsucht wurde, ob
    er nicht etwa an seiner Person die Mittel h�tte, dem Gesetze
    vorzugreifen. Endlich brachten sie ihn in eine der Armes�nderzellen und
    lie�en ihn darin -- allein.

    Er setzte sich der T�r gegen�ber auf eine steinerne Bank, die als Sitz
    und Lager diente, heftete die blutunterlaufenen Augen auf den Boden
    und bem�hte sich, seine Gedanken zu sammeln. Nach einiger Zeit begann
    er sich einzelner Bruchst�cke der Anrede des Richters zu erinnern,
    obwohl es ihm, w�hrend sie gesprochen worden, gewesen war, als wenn
    er kein Wort h�ren k�nnte. Ein Teil f�gte sich allm�hlich zum andern,
    und endlich stand das Ganze fast vollst�ndig klar vor ihm. Aufgeh�ngt
    zu werden am Halse, bis er tot w�re -- das war das Ende gewesen.
    Aufgeh�ngt zu werden, bis er tot w�re.

    Es wurde dunkel, sehr dunkel, und er fing an, aller derer zu gedenken,
    die er gekannt und die auf dem Schafott gestorben waren -- einige durch
    seine Schuld oder auf seinen Betrieb. Sie tauchten in so rascher Folge
    vor ihm auf, da� er sie kaum zu z�hlen vermochte. Er hatte mehrere von
    ihnen sterben sehen -- und sie verspottet, weil sie mit Gebeten auf
    den Lippen verschieden waren. Wie gedankenschnell sie aus starken,
    kr�ftigen M�nnern in baumelnde Fleischklumpen verwandelt waren!

    Mancher von ihnen hatte vielleicht dasselbe Gemach bewohnt -- auf
    derselben Stelle gesessen. Es war sehr finster -- warum wurde kein
    Licht gebracht? Die Zelle war vor vielen Jahren erbaut -- Hunderte
    mu�ten ihre letzten Stunden darin verlebt haben -- man sa� darin wie
    in einem mit Leichen angef�llten Gew�lbe -- und viele derselben hatten
    wohlbekannte Gesichter -- Licht, Licht!

    Endlich, als er sich die H�nde an der eisenverwahrten T�r fast blutig
    geschlagen hatte, erschienen zwei M�nner, deren einer ein Licht trug,
    das er auf einen eisernen, in der Mauer befestigten Leuchter steckte,
    w�hrend der andere eine Matratze hinter sich herzog, um darauf die
    Nacht zuzubringen, denn der Gefangene sollte fortan nicht mehr allein
    gelassen werden.

    Dann kam die Nacht -- die finstere, schauerliche, schweigende Nacht.
    Andere Wachende freuen sich, die Kirchglocken schlagen zu h�ren, die
    vom Leben zeugen und den nahenden Tag verk�nden. Dem Juden brachten sie
    Verzweiflung. Jedes Anschlagen des eisernen Kl�ppels f�hrte ihn zu dem
    einen hohlen Schall -- Tod. Was n�tzte das Ger�usch des gesch�ftigen,
    heiteren Morgens, das selbst in den Kerker und zu ihm drang? Es war
    Totengel�ute anderer Art, das noch den Hohn zur schrecklichernsten
    Mahnung hinzuf�gte.

    Der Tag verging -- Tag! Da war kein Tag; er war so bald entschwunden
    wie angebrochen, und abermals kam die Nacht -- Nacht! So lang und
    doch so kurz; lang in ihrem schrecklichen Schweigen, und kurz nach
    ihren fl�chtigen Stunden. Jetzt redete der Elende irre und stie�
    Gottesl�sterungen aus -- dann heulte er und zerraufte sein Haar.
    Ehrw�rdige M�nner seines Glaubens waren gekommen, mit ihm zu beten,
    allein er hatte sie mit Verw�nschungen hinausgetrieben. Sie erneuerten
    ihre menschenfreundlichen Versuche und mu�ten seinen gewaltt�tigen
    Drohungen weichen.

    Sonnabend -- nur noch eine einzige Nacht! Und w�hrend er noch sann und
    sann: nur noch eine einzige Nacht! d�mmerte es schon -- Sonntag!

    Erst am Abend dieses schauervoll-bangen Tages ward seine verpestete
    Seele von einem vernichtenden Gef�hle ihrer verzweifelten Lage
    ergriffen. Nicht, da� er auch nur von fern eine bestimmte Hoffnung,
    Gnade zu erlangen, gehegt h�tte; er hatte es nur noch nicht �ber sich
    vermocht, den Gedanken, so bald sterben zu m�ssen, klar und deutlich
    auszudenken. Er hatte nur wenig zu den beiden M�nnern gesprochen,
    die sich einander bei ihm abl�sten, und sie hatten sich ihrerseits
    nicht um ihn gek�mmert. Er hatte wachend dagesessen, aber getr�umt.
    Jetzt sprang er von Minute zu Minute auf und rannte mit keuchendem
    Munde und brennender Stirn in entsetzlicher Furcht- und Zorn- und
    Grimmanwandlung auf und nieder, da� sie sogar -- die an dergleichen
    Gew�hnten -- schaudernd vor ihm zur�ckbebten. Er wurde zuletzt unter
    den Folterqualen seines b�sen Gewissens so f�rchterlich, da� keiner es
    ertragen konnte, allein bei ihm zu sitzen und ihn vor Augen zu haben,
    -- da� seine W�rter beschlossen, miteinander Wache bei ihm zu halten.

    Er kauerte auf seinem Steinbette nieder und dachte der Vergangenheit.
    Er war bei seiner Abf�hrung in das Gef�ngnis verwundet worden und
    trug deshalb ein leinenes Tuch um den Kopf. Sein rotes Haar hing auf
    sein blutloses Gesicht herunter; sein Bart war zerzaust und in Knoten
    gedreht; aus seinen Augen leuchtete ein schreckliches Feuer; seine
    ungewaschenen Glieder bebten von dem in ihm brennenden Fieber. Acht
    -- neun, zehn! Wenn man die Glocken vielleicht nicht schlagen lie�,
    blo� um ihn mit Schrecken zu erf�llen, wenn sie die einander auf den
    Fersen folgenden Stunden wirklich anzeigten -- wo mu�te er sein, wenn
    sie abermals schlugen? Elf! Noch ein Schlag, ehe die Stimme der letzten
    Stunde verklungen war. Um acht Uhr war er, wie er sich sagte, der
    einzige Leidtragende zu seinem eigenen Grabgefolge; um elf --

    Newgates schreckliche Mauern, die so viel Elend und so unaussprechliche
    Angst und Pein nicht blo� vor den Augen, sondern nur zu oft und zu
    lange auch vor den Gedanken der Menschen verbargen, umschlossen nie ein
    so entsetzliches Schauspiel wie dieses. Die wenigen Vor�bergehenden,
    die etwa stillstanden und bei sich dachten, was der Verurteilte wohl
    vornehmen m�chte, der am folgenden Tage hingerichtet werden sollte,
    w�rden die Nacht darauf gar schlecht geschlafen haben, wenn sie ihn im
    selben Augenblicke h�tten sehen k�nnen.

    Vom Abend bis fast um Mitternacht traten bald einzelne, bald mehrere
    zu dem Pf�rtner, fragten in gro�er Spannung, ob ein Aufschub der
    Hinrichtung verf�gt sei, und teilten die willkommene Verneinung andern,
    in Haufen Stehenden mit, die auf die T�r hinwiesen, aus welcher er
    kommen m��te, die Stelle zeigten, wo das Schafott errichtet werden
    w�rde, sich widerstrebend entfernten und im Fortgehen die zu erwartende
    Szene sich im voraus ausmalten. Endlich waren alle heimgekehrt und die
    Stra�en umher auf eine Stunde in der Mitte der Nacht der Einsamkeit und
    Finsternis �berlassen.

    Der Raum vor dem Gef�ngnisse war ges�ubert, und man hatte einige
    starke, schwarz bemalte Schranken, dem vorauszusehenden gro�en Gedr�nge
    zu wehren, errichtet, als Mr. Brownlow mit Oliver an dem Pf�rtchen
    erschien und eine Sheriffserlaubnis vorwies, den Verurteilten sehen zu
    d�rfen. Sie wurden sogleich eingelassen.

    �Soll der kleine Herr auch mit hinein, Sir?� fragte der Schlie�er, der
    ihnen zum F�hrer gegeben war. �'s ist kein Anblick f�r Kinder, Sir.�

    �Freilich nicht, mein Freund�, erwiderte Brownlow; �allein was ich bei
    dem Manne zu tun habe, hat auch auf den Knaben sehr genauen Bezug, und
    da er ihn als gl�cklichen Frevler gekannt hat, so halte ich es f�r gut,
    da� er ihn auch jetzt sehe, ob es auch einen etwas peinlichen Eindruck
    bei ihm hervorbringen mag.�

    Die Worte waren leise gesprochen. Der Schlie�er ber�hrte den Hut,
    blickte mit einiger Neugier nach Oliver und ging ihnen voran, zeigte
    ihnen das Tor, aus welchem der Verurteilte kommen w�rde, machte sie
    aufmerksam auf das an ihr Ohr dringende H�mmern der das Schafott
    erbauenden Zimmerleute und �ffnete ihnen endlich die T�r der Zelle des
    Juden.

    Dieser sa� auf seinem Bette, wiegte sich hin und her, und sein Gesicht
    glich mehr dem eines eingefangenen Tieres als einem menschlichen
    Antlitze. Er gedachte offenbar seines alten Lebens, denn er murmelte,
    Brownlow und Oliver sehend und doch nicht sehend, vor sich hin: �Guter
    Junge, Charley -- gemacht brav -- und auch Oliver -- ha, ha, ha, Oliver
    -- und sieht aus wie ein Junker -- ganz wie ein -- bringt ihn fort --
    zu Bett mit dem Buben!�

    Der Schlie�er fa�te Oliver bei der Hand und fl�sterte ihm zu, da� er
    ohne Furcht sein m�chte.

    �Zu Bett mit ihm!� rief der Jude. �H�rt Ihr nicht? Er -- er ist -- ist
    an diesem allen schuld. 's ist des Geldes wert, ihn zu erziehen dazu --
    Bolters Kehle, Bill; k�mmert Euch um die Dirne nicht -- Bolters Kehle,
    so tief Ihr k�nnt schneiden. S�gt ihm ab den Kopf.�

    �Fagin�, sagte der Schlie�er.

    �Ja, ja�, rief der Jude und nahm rasch die lauschende Stellung an, die
    er bei seinem Prozesse behauptet hatte. �Ein alter Mann, Mylord; ein
    sehr, sehr alter Mann.�

    �Hier ist jemand, Fagin, der Euch etwas zu sagen hat -- seid Ihr ein
    Mann?� rief ihm der Schlie�er, ihn sch�ttelnd und dann festhaltend, in
    das Ohr.

    �Werd's nicht mehr sein lange�, rief der Jude zur�ck, mit einem
    Angesicht aufblickend, das keinen menschlichen Ausdruck mehr hatte --
    nur Wut und Entsetzen malte sich darin. �Schlagt sie alle tot! Was
    haben sie f�r ein Recht, mich abzuschlachten?�

    Er erkannte jetzt Oliver und Brownlow, wich in die fernste Ecke
    seines Sitzes zur�ck und fragte, was sie an diesen Ort gef�hrt h�tte.
    Der Schlie�er hielt ihn fortw�hrend fest und forderte Brownlow auf,
    rasch zu sagen, was er ihm zu sagen h�tte, denn er w�rde mit jedem
    Augenblicke schlimmer.

    �Es sind Euch gewisse Papiere zu sicherer Aufbewahrung anvertraut
    worden, und zwar von einem Menschen, namens Monks�, sagte Brownlow,
    sich ihm n�hernd.

    �'s ist gelogen -- ich habe keine, keine, keine!� erwiderte der Jude.

    �Um der Liebe Gottes willen,� sagte Brownlow feierlich, �sprecht nicht
    so am Rande des Grabes, sondern sagt mir, wo ich sie finden kann.
    Ihr wi�t, da� Sikes tot ist, da� Monks gestanden hat, da� Ihr keine
    Hoffnung eines Gewinnes mehr habt. Wo sind die Papiere?�

    �Oliver,� rief der Jude, dem Knaben winkend, �komm, la� mich dir
    fl�stern ins Ohr.�

    �Ich habe keine Furcht�, sagte Oliver leise zu Brownlow und ging zu
    dem Juden, der ihn zu sich zog und ihm zufl�sterte: �Sie sind in 'nem
    leinenen Beutel in 'nem Loche des Schornsteins oben im Vorderzimmer.
    Ich m�chte gern reden mit dir, mein Lieber -- m�chte reden mit dir.�

    �Ja, ja�, erwiderte Oliver. �La�t mich ein Gebet sprechen, betet auf
    Euren Knien mit mir, und wir wollen bis morgen fr�h miteinander reden.�

    �Drau�en, drau�en�, sagte der Jude, den Knaben vor sich nach der T�r
    hindr�ngend und mit einem leeren, starren Blicke �ber seinen Kopf
    schauend. �Sag', ich w�re eingeschlafen -- *dir* werden sie's glauben.
    Du kannst mir helfen 'raus, wenn du tust, was ich dir sage. Jetzt,
    jetzt!�

    �O Gott, verzeihe diesem ungl�cklichen Manne!� rief der Knabe unter
    hervorst�rzenden Tr�nen.

    �So ist's recht, so ist's recht! Das ist das wahre Mittel! Diese T�r
    zuerst. Beb' und zittr' ich, wenn wir am Galgen vor�bergehen, achte
    darauf nicht, sondern mach fort, rasch fort. Jetzt, jetzt, jetzt!�

    �Haben Sie ihm nichts mehr zu sagen, Sir?� fragte der Schlie�er.

    �Nein�, erwiderte Brownlow. �Wenn ich hoffen k�nnte, da� wir ein Gef�hl
    seiner Lage in ihm erwecken --�

    �Das ist unm�glich, Sir�, fiel der Schlie�er kopfsch�ttelnd ein. �Ich
    mu� Ihnen den Rat geben, ihn zu verlassen.�

    Die beiden W�rter kehrten jetzt zur�ck, und der Jude rief: �Fort,
    fort! Tritt leise auf -- aber nicht so langsam. Schneller, schneller!�
    Sie befreiten den Knaben von seinem Griffe und hielten ihn selbst
    zur�ck. Er suchte sich mit der Kraft der Verzweiflung loszumachen und
    stie� einen Schrei nach dem andern aus, der selbst die ellendicken
    Kerkermauern durchdrang und in Brownlows und Olivers Ohren t�nte, bis
    sie in den offenen Hof hinaustraten.

    Sie konnten das Gef�ngnis nicht sogleich verlassen. Oliver war einer
    Ohnmacht nahe und so angegriffen, da� eine Stunde verflo�, ehe er seine
    F��e zu gebrauchen vermochte.

    Der Tag brach an, als sie das Gef�ngnis verlie�en. Es hatte sich
    schon eine gro�e Volksmenge gesammelt: die Fenster waren mit Leuten
    angef�llt, die sich rauchend und Karten spielend die Zeit vertrieben;
    die Untenstehenden dr�ngten sich hin und her, stritten und scherzten
    miteinander. Die ganze Umgebung bot ein heiteres, belebtes Schauspiel
    dar -- in dessen Mitte schauerliche Zur�stungen an Verbrechen, Gericht,
    Strafe und Tod erinnerten.




    53. Kapitel.

        Schlu�.


    Was zu erz�hlen jetzt noch er�brigt, ist in wenigen Worten zu berichten.

    Noch vor dem Ablaufe von drei Monaten wurde Rose Fleming und Harry
    Maylie in der Dorfkirche getraut, welche fortan der Schauplatz der
    T�tigkeit des jungen Geistlichen sein sollte. An demselben Tage nahmen
    sie von ihrer neuen freundlichen Wohnung Besitz. Mrs. Maylie schlug
    ihren Wohnsitz bei ihnen auf, um den Rest ihrer Tage durch die beste
    Freude zu versch�nen, die dem ehrw�rdigen Alter zuteil werden kann --
    den Anblick der Seligkeit der Lieben, deren Bildung und Begl�ckung die
    beste Zeit und die besten Kr�fte eines wohlverlebten Daseins gewidmet
    gewesen sind.

    Monks wie seine Mutter waren mit dem Verm�gen, das sie an sich
    gerissen, so verschwenderisch umgegangen, da� f�r den ersteren und
    Oliver, wenn der Rest unter beide geteilt wurde, nur dreitausend Pfund
    �brigblieben. Nach dem Testament seines Vaters hatte Oliver Anspruch
    auf das ganze; allein Mr. Brownlow schlug eine Teilung vor, um den
    �lteren Bruder der Mittel nicht zu berauben, ein neues und besseres
    Leben zu beginnen, womit sich Oliver von ganzem Herzen zufrieden
    erkl�rte.

    Monks begab sich unter Beibehaltung seines angenommenen Namens in die
    neue Welt, vergeudete rasch das ihm gelassene, beging neue Verbrechen,
    sa� lange im Kerker und erlag darin endlich einem Anfalle seiner alten
    Krankheit. In ebenso weiter Ferne von der Heimat starben die noch
    �brigen Hauptmitglieder der Bande Fagins.

    Mr. Brownlow adoptierte Oliver, bezog mit ihm und Frau Bedwin eine vom
    Pfarrhause nur eine Meile entfernte Wohnung, befriedigte dadurch den
    einzigen noch nicht erf�llten Wunsch des warmen und liebevollen Herzens
    Olivers und half einen kleinen Freundeskreis bilden, in welchem ein so
    vollkommenes Gl�ck herrschte, wie es in dieser ver�nderlichen Welt nur
    zu finden ist.

    Bald nach der Verm�hlung des jungen Paares kehrte der w�rdige Doktor
    nach Chertsey zur�ck, wo er, des Umgangs seiner alten Freunde beraubt,
    wenn sein Temperament dergleichen zugelassen, mi�m�tig geworden
    sein und sich in einen Murrkopf verwandelt haben w�rde, wenn er es
    anzufangen gewu�t h�tte. Nachdem er einige Monate geschwankt, �bertrug
    er seine Praxis seinem Assistenten und siedelte nach dem Wohnorte
    Maylies hin�ber, wo er Gartenbau trieb, pflanzte, fischte, zimmerte
    usw., und zwar alles mit seiner eigent�mlichen Leidenschaftlichkeit,
    so da� er bald in allem, was er trieb, weit und breit umher eine
    bedeutende Autorit�t wurde.

    Er hatte eine gro�e Freundschaft f�r Mr. Grimwig gefa�t, welche von
    dem exzentrischen Gentleman mit ebenso gro�er W�rme erwidert wurde.
    Grimwig besucht ihn daher h�ufig und pflanzt, fischt und zimmert mit,
    jedoch stets auf eine eigent�mliche und bislang unbekannte Weise;
    er behauptet indes stets bei seiner Lieblingsbeteuerung, da� es die
    richtige sei. An Sonntagen verfehlt er nie, dem jungen Geistlichen in
    das Angesicht die Predigt zu kritisieren und versichert Mr. Losberne
    hinterher im strengsten Vertrauen, sie w�re nach seinem Urteile eine
    ganz vortreffliche Arbeit gewesen, er hielte es indes f�r gut, nichts
    davon zu sagen. Es ist eine stehende und gro�e Lieblingsbelustigung
    Mr. Brownlows, ihn mit seiner alten, Oliver betreffenden Prophezeiung
    aufzuziehen und an den Abend zu erinnern, an welchem sie die Uhr
    auf den zwischen ihnen stehenden Tisch gelegt hatten und des Knaben
    R�ckkehr erwarteten; allein Grimwig erkl�rte dann ohne Ausnahme, da�
    er in der Hauptsache doch recht gehabt habe, denn Oliver w�re eben
    nicht zur�ckgekommen, eine Bemerkung, welche von seiner Seite jedesmal
    belacht wird, was seine gute Laune noch verbessert.

    Mr. Claypole wurde begnadigt, weil er wider den Juden als Zeuge
    aufgetreten, erachtete aber sein Handwerk nicht f�r so sicher, wie
    er es wohl w�nschen mochte, und war eine Weile in Verlegenheit, wie
    er ohne zuviel Arbeit seinen Lebensunterhalt gewinnen sollte. Er hat
    nach reiflicher �berlegung das Gesch�ft eines Angebers begonnen, das
    ihn sehr anst�ndig ern�hrt. Er geht n�mlich Sonntags w�hrend des
    Gottesdienstes mit Charlotte w�rdevoll gekleidet aus. Die Dame sinkt
    an den T�ren menschenfreundlicher Wirte in Ohnmacht, der Herr l��t
    Branntwein f�r sie geben, um sie wieder ins Bewu�tsein zur�ckzurufen,
    bringt am folgenden Tage die Sabbatsverletzung zur Anzeige und steckt
    die H�lfte der Strafe ein, welche der Wirt bezahlen mu�. Bisweilen wird
    Mr. Claypole selbst ohnm�chtig, das Ergebnis ist aber dasselbe.

    Mr. und Mrs. Bumble versanken, ihrer Stellen beraubt, allm�hlich in
    gro�es Elend und D�rftigkeit und wurden endlich als Arme in dasselbe
    Verpflegungshaus des Kirchspiels aufgenommen, in welchem sie einst
    geherrscht hatten. Man hat Mr. Bumble sagen h�ren, da� er bei dieser
    Umkehr und Erniedrigung nicht einmal Mut und Lust habe, f�r die
    Trennung von seiner Frau dankbar zu sein.

    Mr. Giles und Brittles bekleiden fortw�hrend ihre alten �mter
    und W�rden; nur ist der erstere kahl und der letztgenannte Knabe
    vollkommen grau geworden. Sie schlafen im Pfarrhause, widmen aber
    ihre Aufmerksamkeiten den Bewohnern desselben, Oliver, Brownlow
    und Losberne, so gleichm��ig, da� die Leute im Dorfe niemals haben
    erforschen k�nnen, wem sie eigentlich dienen.

    Master Charley Bates, ersch�ttert durch Sikes' Verbrechen, geriet auf
    den Gedanken, ob ein rechtschaffenes Leben nicht am Ende doch noch
    das beste w�re, �berlegte, kam zu dem Schlusse, da� dem so sei, und
    nahm sich vor, den Pfad der Tugend zu erw�hlen. Es wurde ihm eine
    Zeitlang �u�erst schwer, er litt nicht wenig dabei, allein es gelang
    ihm endlich, da er einen zufriedenen und festen Sinn besa�. Er ging in
    saure Dienste bei einem P�chter, darauf bei einem Fuhrmanne und ist
    gegenw�rtig der munterste junge Viehh�ndler in ganz Northamptonshire.

    Und nun, am Schlusse, beginnt mir die Hand, welche dies niederschreibt,
    zu beben, und gern sp�nne ich den Faden meiner Erz�hlung noch ein
    wenig l�nger aus -- verweilte so gern noch bei einigen der mir teuer
    Gewordenen, in deren geistigem Umgange ich mich so lange bewegt,
    um ihr Gl�ck durch den Versuch seiner Schilderung zu teilen. Ich
    m�chte Rose Maylie in der ganzen Bl�te und Anmut der jungen Gattin
    schildern, wie sie auf ihren von der gro�en Welt entfernten Lebenspfad
    ein so mildes und sch�nes Licht warf, das auf alle mit ihr ihn
    Wandelnde fiel und in ihre Herzen leuchtete; -- ich m�chte sie als
    das Leben und die Lust des traulichen Kreises am Kamine und der froh
    in der Sommerlaube Versammelten schildern; ich m�chte ihr Mittags
    im Sonnenglanze folgen und den sanften Ton ihrer s��en Stimme bei
    Spazierg�ngen an den mondhellen Abenden vernehmen; sie bei ihren
    stillen Wohlt�tigkeitswanderungen und im Hause beobachten, wie sie
    l�chelnd und unerm�det ihre h�uslichen Pflichten erf�llt; m�chte ihr
    Gl�ck und das des Kindes ihrer hin�bergegangenen Schwester malen, das
    sie genossen in gegenseitiger Liebe, in wehm�tig-s��en Gedanken an so
    traurig verlorene Teure; m�chte vor mir die fr�hlich sie umspielenden,
    munter-geschw�tzigen Kleinen hinzaubern; m�chte mir den Ton ihres
    frohen Gel�chters, die Freudentr�ne in ihrem sanften blauen Auge --
    ihr holdes L�cheln, ihre verst�ndige Rede -- jeden Blick, jedes Wort
    zur�ckrufen.

    Wie Mr. Brownlow seinen angenommenen Sohn von einem Fortschritte in
    Kenntnissen aller Art zum andern f�hrte und ihn, je mehr er sich
    entwickelte, immer lieber gewann -- wie er in seinem Antlitze die Z�ge
    der Geliebten seiner Jugend suchte und mehr und mehr fand -- wie sich
    die beiden durch Mi�geschick gepr�ften Waisen der Lehren desselben
    erinnerten und sie durch Milde und Nachsicht und Liebe gegen andere
    �bten und unter inbr�nstigem Danke gegen den Gott, der sie besch�tzt
    und gerettet -- das alles braucht nicht erz�hlt zu werden; denn ich
    habe gesagt, da� sie wahrhaft gl�cklich waren, und ohne echte, innige
    Menschenliebe, ohne Dankbarkeit gegen ihn im Herzen, dessen Gesetzbuch
    Gnade hei�t und Erbarmen, und der die Liebe selbst ist gegen alles, was
    Odem hat, kann wahres Gl�ck nimmer gewonnen werden.

    Neben dem Altare der alten Dorfkirche erblickt man eine wei�e
    Marmortafel, auf welcher nur das eine Wort -- �Agnes!� eingegraben ist.
    In dem Grabgew�lbe darunter befindet sich ein Sarg, und m�chten noch
    viele, viele Jahre vergehen, ehe ein zweiter Name hinzugef�gt wird!
    Doch wenn die Geister der Toten zur Erde zur�ckkehren, die durch Liebe
    -- �ber das Grab hinausreichende Liebe geheiligten St�tten zu besuchen
    -- Wohnst�tten derer, die sie in ihrem Leben kannten, so glaube ich,
    da� der Schatten des armen M�dchens oft, oft das leere Pl�tzchen
    umschwebt, obwohl es sich in einer Kirche befindet, und obwohl Agnes
    schwach war und vom rechten Pfade abirrte.

      [Illustration]




    Ber�hmte Romane

    und Werke der Weltliteratur


    Serie _I_

        =Alexis=, Die Hosen des Herrn von Bredow. Roman.

        =Anzengruber=, Der Schandfleck. Eine Dorfgeschichte.

        =Auerbach=, Barf��ele. Eine Dorfgeschichte.

        =Brachvogel=, Friedemann Bach. Roman.

        --, Das R�tsel von Hildburghausen. Roman.

        =Bulwer=, Die letzten Tage von Pompeji. Histor. Roman.

        =Casanova=, Memoiren.

        =Daudet=, Fromont jr. und Risler sen. Roman.

        =Dickens=, Oliver Twist. Roman.

        =Hauff=, Lichtenstein. Romantische Sage.

        =Immermann=, Der Oberhof.

        =Jacobsen=, Niels Lyhne. Roman.

        =Lagerl�f=, G�sta Berlings Geschichte.

        =Ludwig=, Zwischen Himmel und Erde.

        =Maupassant=, Der Frauenliebling (_Bel-Ami_).

        --, Zur linken Hand. Novellen.

        =Pr�vost=, Manon Lescaut. Die Geschichte einer Liebe.

        =Scheffel=, Ekkehard. Eine Geschichte aus dem zehnten Jahrhundert.

        =Scott=, Ivanhoe. Roman.

        =Sienkiewicz=, Strudel. Roman.

        =Storm=, Immensee und andere Novellen.

        --, Der Schimmelreiter und andere Novellen.

        =Wilde=, Das Bildnis des Dorian Gray.

        =Zoozmann=, La�t Uns Lachen!


        Globus Verlag G. m. b. H. Berlin




    Ber�hmte Romane

    und Werke der Weltliteratur


    Serie _II_

        =Dumas=, Der Graf von Monte Christo. Vollst�ndige Ausgabe. 2 Bde.

        =Freytag=, Die Ahnen. Vollst�ndige Ausgabe. 3 Bde.

        =Lagerl�f=, Jerusalem.

        =Samarow=, Die Saxoborussen. Roman.

        =Sienkiewicz=, Die Familie Polaniecki. Roman.

        --, Die Kreuzritter. Historischer Roman.

        --, Ohne Dogma. Roman.

        --, Quo vadis? Historischer Roman.

        =Wallace=, Ben Hur. Erz�hlung aus der Zeit Christi.

        Globus Verlag G. m. b. H. Berlin

    Hermann Schmidt's Buch- und Kunstdruckerei. G. m. b. H., Berlin _O_ 27.




    Anmerkungen zur Transkription:

    In "Sikes zuckte ungeduldig die Achseln, als wenn er die Vorsicht f�r
    �berfl�ssig hielte," stand "Achsel".

    In "Und wenn eine Rotte von Teufeln seine Gestalt ann�hme," stand
    "annehme"

    In "In welchem der Leser, wenn er in das sechsunddrei�igste Kapitel
    zur�ckblicken will, einen im ehelichen Leben nicht selten
    hervortretenden Kontrast beobachten wird." stand "ehrlichen"

    In "Wenn Sie in bed�rftiger Lage oder sonst ungl�cklich sind," fehlte
    das "in"

        

    
    `;
    console.log('Textdaten geladen.');
    return textData;
}